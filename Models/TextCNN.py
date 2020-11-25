# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self,data_dir):
        self.model_name = 'TextCNN'
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir + 'test.txt'
        self.dev_path = data_dir + 'val.txt'
        self.class_list = ['0','1','2','3','4','5','6','7','8','9']  # 类别名单

        self.embedding_pretrained = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.num_classes = len(self.class_list)
        self.require_improvement = 5000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 70  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.max_len = 100  # 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5  # 学习率
        self.embed = 300  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.tokenizer = BertTokenizer.from_pretrained('./vocab/')
        self.n_vocab = 21128
        self.pred_path = './data/prediction.csv'

'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
