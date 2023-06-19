

from transformers import BertTokenizer, BertForQuestionAnswering
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt_MRCModel(nn.Module):
    def __init__(self, model_name, tokenizer, max_topic_len=35, max_seq_len = 256):
        super(Prompt_MRCModel, self).__init__()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.topic_model = BertForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.max_topic_len = max_topic_len
    
    
    def generate_default_inputs(self, batch, topic_embed, device):

        input_ids = batch['input_ids']
        bz = batch['input_ids'].shape[0]
        block_flag = 1
        raw_embeds = self.model.bert.embeddings.word_embeddings(input_ids.to(device)).squeeze(1)
        topic_embeds = self.topic_model.bert.embeddings.word_embeddings(topic_embed.to(device)).squeeze(1)
        input_embeds = torch.cat((topic_embeds,raw_embeds),1)
                
        inputs = {'inputs_embeds': raw_embeds.to(device), 'attention_mask': batch['attention_mask'].squeeze(1).to(device)}
        inputs['token_type_ids'] = batch['token_type_ids'].squeeze(1).to(device)
        return inputs

    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):

        return self.model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask,
                          labels=labels,
                          token_type_ids=token_type_ids)

    def mlm_train_step(self, batch, topic_embed, start_positions, end_positions, device):

        inputs_prompt = self.generate_default_inputs(batch, topic_embed, device)
        bert_out = self.model(**inputs_prompt, start_positions=start_positions, end_positions=end_positions)
        return bert_out
    
    
