# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

import json


class PATENTDataset(Dataset):
    def __init__(self, corpus='patent', split='train', max_length=512, padding='max_length'):
        self.split = split
        self.max_length = max_length
        # self.tokenizer = AutoTokenizer.from_pretrained('chinese-roberta-large')
        self.tokenizer = AutoTokenizer.from_pretrained('../model/chinese-roberta-large')
        # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.filename = split + '.json'
        self.corpus = self.read_corpus()

    def read_corpus(self):
        t = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                dic = json.loads(line)
                t.append(dic)
        return t

    def __getitem__(self, item):
        ins = self.corpus[item]
        id, title, assignee, abstract, label_id = ins['id'], ins['title'], ins['assignee'], ins['abstract'], ins[
            'label_id']
        label_id = label_id
        input = "@标题：" + title + "@专利授权人：" + assignee + "@专利内容：" + abstract
        input_id = self.tokenizer(input)
        input_ids = {}
        for key, value in input_id.items():
            input_ids[key] = torch.tensor([value])
        if input_ids['input_ids'].shape[-1] >= self.max_length:
            input_ids['input_ids'] = input_ids['input_ids'][:, :self.max_length]
            input_ids['token_type_ids'] = input_ids['token_type_ids'][:, :self.max_length]
            input_ids['attention_mask'] = input_ids['attention_mask'][:, :self.max_length]
        else:
            need_pad = self.max_length - input_ids['input_ids'].shape[-1]
            input_ids_pad = torch.full([1, need_pad], self.tokenizer.pad_token_id, dtype=torch.long)
            token_type_ids_pad = torch.full([1, need_pad], 0, dtype=torch.long)
            attention_mask_pad = torch.full([1, need_pad], 0, dtype=torch.long)

            input_ids['input_ids'] = torch.cat((input_ids['input_ids'], input_ids_pad), dim=-1)
            input_ids['token_type_ids'] = torch.cat((input_ids['token_type_ids'], token_type_ids_pad), dim=-1)
            input_ids['attention_mask'] = torch.cat((input_ids['attention_mask'], attention_mask_pad), dim=-1)
        inputs = {
            'input_ids': torch.squeeze(input_ids['input_ids']),
            'token_type_ids': torch.squeeze(input_ids['token_type_ids']),
            "attention_mask": torch.squeeze(input_ids['attention_mask']),
            "labels": torch.nn.functional.one_hot(torch.tensor([label_id]), num_classes=36).to(torch.float)
        }
        return inputs

    def __len__(self):
        return len(self.corpus)