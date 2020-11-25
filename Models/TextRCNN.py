# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self,data_dir):
        self.model_name = 'TextRCNN'
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir + 'test.txt'
        self.dev_path = data_dir + 'val.txt'
        self.class_list = ['0','1','2','3','4','5','6','7','8','9']             # 类别名单                           # 词表
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.embedding_pretrained = None                                   # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 1.0                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                      # 类别数
        self.tokenizer = BertTokenizer.from_pretrained('./vocab/')
        self.n_vocab = 21128
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 40                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                       # 学习率
        self.embed = 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数

        self.pred_path = './data/prediction.csv'

'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
