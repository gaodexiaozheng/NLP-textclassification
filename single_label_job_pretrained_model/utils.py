# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from torch.nn import functional as F
from sklearn import metrics
PAD, CLS,SEP = '[PAD]', '[CLS]','[SEP]'  # padding符号, bert中综合信息符号
import numpy as np
import pandas as pd

def build_dataset(config):
    def load_dataset(path, max_len=40):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('[SEP]')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if max_len:
                    if len(token) < max_len:
                        mask = [1] * len(token_ids)
                    else:
                        mask = [1] * max_len
                        token_ids = token_ids[:max_len]
                        seq_len = max_len
                contents.append([token_ids, config.class_list.index(label), seq_len, mask])
        return contents
    train = load_dataset(config.train_path, config.max_len)
    dev = load_dataset(config.dev_path, config.max_len)
    test = load_dataset(config.test_path, config.max_len)
    return train, dev, test

def build_predict_dataset(config):

    contents = []
    data = pd.read_table(config.test_path)
    for i, row in data.iterrows():
        token = config.tokenizer.tokenize(row['sentence'])
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        if config.max_len:
            if len(token) < config.max_len:
                mask = [1] * len(token_ids)
            else:
                mask = [1] * config.max_len
                token_ids = token_ids[:config.max_len]
                seq_len = config.max_len
        contents.append([token_ids, seq_len, mask])

    return contents,data

def pad_batch_data(datas):
    batch_length = max([_[2] for _ in datas])

    for i in range(len(datas)):
        if len(datas[i][0]) < batch_length:
            datas[i][0] += ([0] * (batch_length- datas[i][2]))
            datas[i][-1] += ([0] * (batch_length- datas[i][2]))
    return datas

class PreDatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        datas = pad_batch_data(datas)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        mask[:,[0]] = 2
        return (x, seq_len, mask)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def pred_iterator(dataset, config):
    iter = PreDatasetIterater(dataset, config.batch_size, config.device)
    return iter
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        datas = pad_batch_data(datas)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        mask[:,[0]] = 2
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter




def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))