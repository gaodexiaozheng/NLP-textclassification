# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer
from single_label_job_normal_model.utils import loss_fn
import os
class Config(object):

    """配置参数"""
    def __init__(self,data_dir):
        self.model_name = 'DPCNN'                                # 测试集
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir + 'test.txt'
        self.dev_path = data_dir + 'dev.txt'
        self.class_list = ['0','1','2','3','4','5','6','7','8','9']        # 类别名单
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.embedding_pretrained = None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                     # 类别数
        self.n_vocab = 21128                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.max_len = 512                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                      # 学习率
        self.embed = 300           # 字向量维度
        self.num_filters = 256     # 卷积核数量(channels数)
        self.pred_path = './data/prediction.csv'

        self.tokenizer = BertTokenizer.from_pretrained('./vocab/')
        self.loss_fn =loss_fn



'''Deep Pyramid Convolutional Neural Networks for Text Categorization'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = x[0]
        batch = x.size()[0]
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        px = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(px)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = px + x # residual
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = x.view(batch,-1)
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
