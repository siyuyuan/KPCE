

import torch
from pymongo import MongoClient
import re
import pickle
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

pad_id = None
text_len_flexible = True
text_len_stable = 50


def device_info(device):
    result = "cpu"
    if torch.cuda.is_available():
        counter = torch.cuda.device_count()
        print("There are {} GPU(s) is available.".format(counter))
        for i in range(counter):
            print("GPU {} Name:{}".format(i, torch.cuda.get_device_name(i)))
        if device == "gpu":
            result = "cuda:0"
            print("We will use {}".format(result))
    return result


def model_paramters_num(model):
    result = 0
    parameters = model.parameters()
    for paramter in parameters:
        result += paramter.numel()
    return result



def load_pickle(path):
    with open(path, 'rb') as fil:
        data = pickle.load(fil)
    return data


def save_pickle(en, path):
    with open(path, 'wb') as fil:
        pickle.dump(en, fil)


class ClassDataset(Dataset):
    def __init__(self, data, tokenizer, max_topic_len=5, max_ques_len=35, max_seq_len = 256):
        super(ClassDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_ques_len = max_ques_len
        self.max_topic_len = max_topic_len
        self.max_seq_len = max_seq_len

    def __len__(self):  
        return len(self.data)
    def find_index(self, big_ls, small_ls):
        s_l = len(small_ls)
        b_l = len(big_ls)
        for i in range(b_l):
            if big_ls[i] == small_ls[0]:
                if big_ls[i:i+s_l] == small_ls:
                    break
        return i


    def __getitem__(self, idx):  

        sample = self.data[idx]
        info = sample[0]
        start_position = sample[1].split(',').index('1')
        end_position = sample[2].split(',').index('1')
        ori_answer = info[start_position:end_position+1]
        ori_answer_encode = self.tokenizer.encode(ori_answer)[1:-1]

        info_text_ids = []
        info_text_ids.extend(self.tokenizer.encode(info)[1:-1])

        start = torch.tensor([self.find_index(info_text_ids, ori_answer_encode) + self.max_ques_len + 2])
        end = torch.tensor([self.find_index(info_text_ids, ori_answer_encode) + len(ori_answer_encode) - 1 + self.max_ques_len + 2])



        if len(info_text_ids) <= self.max_seq_len:
            info_text_ids.extend(
                [self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_seq_len - len(info_text_ids))])


        ques = sample[3].replace('的概念是什么？', '')
        ques_text_ids = []
        ques_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        topic_text_ids = []
        topic_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        if len(ques_text_ids)  <= self.max_ques_len:
            ques_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_ques_len - len(ques_text_ids))])
        if len(topic_text_ids)  <= self.max_topic_len:
            topic_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_topic_len - len(topic_text_ids))])


        ques_embed = torch.tensor(ques_text_ids[:self.max_ques_len])
        info_embed = torch.tensor(info_text_ids[:self.max_seq_len])
        topic_embed = torch.tensor(topic_text_ids[:self.max_topic_len])

        inputs = self.tokenizer(self.tokenizer.decode(ques_text_ids[:self.max_ques_len]),
                                self.tokenizer.decode(info_text_ids[:self.max_seq_len]),
                                return_tensors = 'pt')
        
        return inputs, topic_embed, info_embed, start, end

class Test_ClassDataset(Dataset):
    def __init__(self, data, tokenizer, max_topic_len=5, max_ques_len=35, max_seq_len = 256):
        super(Test_ClassDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_ques_len = max_ques_len
        self.max_topic_len = max_topic_len
        self.max_seq_len = max_seq_len

    def __len__(self):  
        return len(self.data)
    def find_index(self, big_ls, small_ls):
        s_l = len(small_ls)
        b_l = len(big_ls)
        for i in range(b_l):
            if big_ls[i] == small_ls[0]:
                if big_ls[i:i+s_l] == small_ls:
                    break
        return i


    def __getitem__(self, idx):  

        sample = self.data[idx]
        info = sample[0]
        start = torch.tensor([ 0 + self.max_ques_len + 2])
        end = torch.tensor([0  + self.max_ques_len + 2])
        #ori_answer = info[start_position:end_position+1]
        #ori_answer_encode = self.tokenizer.encode(ori_answer)[1:-1]

        info_text_ids = []
        info_text_ids.extend(self.tokenizer.encode(info)[1:-1])

        #start = torch.tensor([self.find_index(info_text_ids, ori_answer_encode) + self.max_ques_len + 2])
        #end = torch.tensor([self.find_index(info_text_ids, ori_answer_encode) + len(ori_answer_encode) - 1 + self.max_ques_len + 2])



        if len(info_text_ids) <= self.max_seq_len:
            info_text_ids.extend(
                [self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_seq_len - len(info_text_ids))])


        #start = torch.tensor([ find_index(self, big_ls, small_ls) + self.max_ques_len + 2])
        #end = torch.tensor([ find_index(self, big_ls, small_ls) + len()+ self.max_ques_len + 2])
        ques = sample[1]
        ques_text_ids = []
        ques_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        topic_text_ids = []
        topic_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        if len(ques_text_ids)  <= self.max_ques_len:
            ques_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_ques_len - len(ques_text_ids))])
        if len(topic_text_ids)  <= self.max_topic_len:
            topic_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_topic_len - len(topic_text_ids))])


        ques_embed = torch.tensor(ques_text_ids[:self.max_ques_len])
        info_embed = torch.tensor(info_text_ids[:self.max_seq_len])
        topic_embed = torch.tensor(topic_text_ids[:self.max_topic_len])

        inputs = self.tokenizer(self.tokenizer.decode(ques_text_ids[:self.max_ques_len]),
                                self.tokenizer.decode(info_text_ids[:self.max_seq_len]),
                                return_tensors = 'pt')
        
        return inputs, topic_embed, info_embed, start, end
    
class Meituan_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_topic_len=5, max_ques_len=35, max_seq_len = 256):
        super(Meituan_Dataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_ques_len = max_ques_len
        self.max_topic_len = max_topic_len
        self.max_seq_len = max_seq_len

    def __len__(self):  
        return len(self.data)

    def __getitem__(self, idx):  

        sample = self.data[idx]
        info = sample[1]
        start = torch.tensor([ 0 + self.max_ques_len + 2])
        end = torch.tensor([0  + self.max_ques_len + 2])
        ques = sample[2].replace('对应的概念是什么？', '')
        ques_text_ids = []
        ques_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        topic_text_ids = []
        topic_text_ids.extend(self.tokenizer.encode(ques)[1:-1])
        if len(ques_text_ids)  <= self.max_ques_len:
            ques_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_ques_len - len(ques_text_ids))])
        if len(topic_text_ids)  <= self.max_topic_len:
            topic_text_ids.extend([self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_topic_len - len(topic_text_ids))])

        info_text_ids = []
        info_text_ids.extend(self.tokenizer.encode(info)[1:-1])
        if len(info_text_ids)  <= self.max_seq_len:
            info_text_ids.extend(
                [self.tokenizer.encode('[PAD]')[1:-1][0] for i in range(self.max_seq_len  - len(info_text_ids))])

        ques_embed = torch.tensor(ques_text_ids[:self.max_ques_len])
        info_embed = torch.tensor(info_text_ids[:self.max_seq_len])
        topic_embed = torch.tensor(topic_text_ids[:self.max_topic_len])

        inputs = self.tokenizer(self.tokenizer.decode(ques_text_ids[:self.max_ques_len]),
                                self.tokenizer.decode(info_text_ids[:self.max_seq_len]),
                                return_tensors = 'pt')
        
        return inputs, topic_embed, info_embed, start, end