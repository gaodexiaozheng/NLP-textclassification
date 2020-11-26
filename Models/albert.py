import sys

from Models.albertmodel import tokenization_albert
from Models.albertmodel.modeling_albert import AlbertModel,AlbertConfig
# from transformers import AlbertModel,AlbertTokenizer
import torch
from torch import nn
class Config(object):
    def __init__(self,data_dir):
        self.model_name = 'albert'
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir +'test.txt'
        self.dev_path = data_dir + 'val.txt'
        self.save_path = './save_model/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 512
        self.require_improvement = 1000
        self.num_epochs = 3
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.bert_path = './pretrained_model_weight/albert/'

        self.tokenizer = tokenization_albert.FullTokenizer(vocab_file=self.bert_path+'vocab.txt',
                                                           do_lower_case=True,
                                                           spm_model_file=None)
        self.hidden_size = 768
        self.class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.num_class = len(self.class_list)


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        configs = AlbertConfig.from_pretrained(config.bert_path+'config.json')
        self.albert = AlbertModel.from_pretrained(config.bert_path+'pytorch_model.bin',from_tf=False,config=configs)
        # self.albert = AlbertModel.from_pretrained(config.bert_path)
        for param in self.albert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size,config.num_class)

    def forward(self,x):
        context = x[0]
        mask  = x[2]
        _,pooled = self.albert(context,attention_mask=mask)
        out = self.fc(pooled)
        return out
