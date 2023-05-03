

import numpy as np
import torch
import random
from transformers import BertTokenizer, BertForQuestionAnswering
import transformers
import logging
import CONFIG
import os
import sklearn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import model_selection
from datetime import datetime
from copy import deepcopy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import json
import argparse
import ysy_util
from ysy_util import ClassDataset
from model import Prompt_MRCModel

logger = None
pad_id = 0
PAD = '[PAD]'


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def create_logger(log_path):
    """
       将日志输出到日志文件和控制台
       """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger

def create_model(vocab_size, model_config_path, pretrained_model_path=None):
    model = BertForQuestionAnswering.from_pretrained(pretrained_model_path)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model

def read_data(data_dir):
    dataset = []
    with open(data_dir, 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, start, end, ques = data.replace('\n','').split('\t')
            dataset.append([info, start, end, ques])
    f.close()
    return dataset

def train(model, train_dataset, train_loader, device, cfg, tokenizer):
    model.train()
    total_steps = int(train_dataset.__len__() * cfg.epochs / cfg.batch_size)
    logger.info("We will process {} steps.".format(total_steps))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.0},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
    ]

    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=cfg.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)
    
      
    
    logger.info("starting training.")
    overall_step = 0
    for epoch in range(cfg.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, sample in enumerate(train_loader):
            inputs = sample[0]
            ques_embed = sample[1].to(device)
            info_embed = sample[2].to(device)
            start = sample[3].to(device)
            end = sample[4].to(device)
            outputs= model.mlm_train_step(inputs, ques_embed, start, end, device)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if (batch_idx + 1) % 2 == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                overall_step += 1
            if (overall_step + 1) % cfg.log_step == 0:
                logger.info(
                    "batch {}/{} of epoch {}/{}, loss {}".format(batch_idx + 1, train_loader.__len__(),
                                                                              epoch + 1, cfg.epochs, loss.item()))
        model_path = os.path.join(cfg.saved_model_path, 'model_epoch{}'.format(epoch + 1))

        logger.info("epoch {} finished.".format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if cfg.save_mode:
            torch.save(model.state_dict(), model_path + '/Prompt_MRCModel.pth')
        epoch_finish_time = datetime.now()
        logger.info("time for one epoch : {}".format(epoch_finish_time - epoch_start_time))
    logger.info("finished train")

def answer_top_k(inputs, ques_embed, info_embed, start, end, tokenizer):
    start_ls = start.cpu().detach().numpy()[0].tolist()
    end_ls = end.cpu().detach().numpy()[0].tolist()
    text_encode = inputs['input_ids'].squeeze()
    l = text_encode.size(0)
    q_l = ques_embed.size(1)
    info_l = info_embed.size(1)
    logit_ls = {}
    for i in range(l):
        for j in range(l):
            if i <= j and i >= 1+q_l+1 and j < l-1:
                text_decode = text_encode.cpu().detach().numpy().tolist()
                if text_decode[j] == 0:
                    break
                else:
                    logit_ls[start_ls[i]+end_ls[j]] = [start_ls[i], end_ls[j], tokenizer.decode(text_decode[i:j+1]).replace(" ", "")]
    result = sorted(logit_ls.items(),key=lambda x:x[0],reverse=True)
    if len(result) > 20:
        return result[:20]
    else:
        return result

def test(model, test_dataset, test_loader, device, cfg, tokenizer):
    with torch.no_grad():
        result = {}
        for batch_idx, sample in enumerate(test_loader):
            inputs = sample[0]
            ques_embed = sample[1].to(device)
            info_embed = sample[2].to(device)
            start = sample[3].to(device)
            end = sample[4].to(device)
            outputs = model.mlm_train_step(inputs, ques_embed, start, end, device)

            result[batch_idx] = []
            answer_texts = answer_top_k(inputs,ques_embed,info_embed,
                                        outputs.start_logits, 
                                        outputs.end_logits,
                                        tokenizer)
            for answer_text in answer_texts:
                dic = {"text": answer_text[1][2], 
                       "probability": answer_text[0], 
                       "start_logit": answer_text[1][0], 
                       "end_logit": answer_text[1][1]}
                
                result[batch_idx].append(dic)

    json_str = json.dumps(result, indent=4)
    with open('result.json', 'w') as json_file:
        json_file.write(json_str)

    
    
def main():
    global logger
    cfg = CONFIG.CONFIG()
    device = ysy_util.device_info(cfg.device)
    print(device)
    logger = create_logger(cfg.log_path)
    tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer_path)
    vocab_size = len(tokenizer)
    model = Prompt_MRCModel(cfg.pretrained_model_path, tokenizer)
    model.to(device)
    n_ctx = cfg.n_ctx
    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    ysy_util.pad_id = pad_id
    if not os.path.exists(cfg.saved_model_path):
        logger.info("build mkdir {}".format(cfg.saved_model_path))
        os.mkdir(cfg.saved_model_path)
    num_parameters = ysy_util.model_paramters_num(model)
    logger.info("number of model parameters:{}".format(num_parameters))
    
    train_set = read_data(cfg.train_data_path)
    dev_set = read_data(cfg.dev_data_path)
    test_set = read_data(cfg.test_data_path)
    
    train_dataset = ClassDataset(train_set,tokenizer)
    valid_dataset = ClassDataset(dev_set,tokenizer)
    test_dataset = ClassDataset(test_set,tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, 
                              shuffle = True, num_workers = cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size = cfg.batch_size, 
                              shuffle = True, num_workers = cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size = 1, 
                             shuffle = False, num_workers = cfg.num_workers)
    train(model, train_dataset, train_loader, device, cfg, tokenizer)
    test(model, test_dataset, test_loader, device, cfg, tokenizer)

os.environ["CUDA_VISIBLE_DEVICES"]="1"
if __name__ == "__main__":
    print("hll")
    main()
