# coding: UTF-8
import os
import torch
import time
from datetime import timedelta
from tqdm import tqdm
from torch.nn import functional as F
import pandas as pd
MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '[UNK]', '[PAD]'  # 未知字，padding符号




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
                seq_len = len(token)
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if max_len:
                    if len(token) < max_len:
                        pass
                    else:
                        token_ids = token_ids[:max_len]
                        seq_len = max_len
                contents.append([token_ids, config.class_list.index(label), seq_len])
        return contents

    train = load_dataset(config.train_path, config.max_len)
    dev = load_dataset(config.dev_path, config.max_len)
    test = load_dataset(config.test_path, config.max_len)
    return train, dev, test

def build_predict_dataset(config):

    contents = []
    data = pd.read_csv(config.pred_path)
    for i, row in data.iterrows():

        token = config.tokenizer.tokenize(row['sentence'])
        seq_len = len(token)
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        if config.max_len:
            if len(token) < config.max_len:
                pass
            else:
                token_ids = token_ids[:config.max_len]
                seq_len = config.max_len
        contents.append([token_ids, seq_len])

    return contents,data

def pred_pad_batch_data(datas):
    batch_length = 512
    for i in range(len(datas)):
        if len(datas[i][0]) < batch_length:
            datas[i][0] += ([0] * (batch_length - datas[i][1]))
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
        datas = pred_pad_batch_data(datas)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        return (x, seq_len)

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
def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def pad_batch_data(datas):
    # batch_length = max([_[2] for _ in datas])
    batch_length = 512
    for i in range(len(datas)):
        if len(datas[i][0]) < batch_length:
            datas[i][0] += ([0] * (batch_length- datas[i][2]))
    return datas

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, flag):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size

        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True

        self.index = 0
        self.device = device
        self.flag = flag

    def _to_n_tensor(self, datas):
        datas = pad_batch_data(datas)
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def _to_f_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            if self.flag == 'normal':
                batches = self._to_n_tensor(batches)
            else:
                batches = self._to_f_tensor(batches)

            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            if self.flag == 'normal':
                batches = self._to_n_tensor(batches)
            else:
                batches = self._to_f_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, flag):
    iter = DatasetIterater(dataset, config.batch_size, config.device, flag)
    return iter


def loss_fn(name):
    if name == 'crossentropy':
        return F.cross_entropy
    elif name == 'focalloss':
        return focal_loss

def onehot(target,class_num=2):
    batch_num = target.size(0)
    labels = torch.full(size=(batch_num, class_num), fill_value=0).cuda()
    labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)
    return labels
def focal_loss(y_pre,y_true):
    alpha = torch.cuda.FloatTensor([0.25,0.75])
    gamma = 2
    y_true = onehot(y_true,2)
    input_soft = F.softmax(y_pre, dim=1) + 1e-8
    weight = torch.pow(1. - input_soft, gamma)
    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(y_true * focal, dim=1)
    return loss_tmp.mean()




def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))





