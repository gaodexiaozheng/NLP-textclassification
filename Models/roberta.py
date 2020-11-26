from transformers import RobertaModel,BertTokenizer
import torch
from torch import nn
class Config(object):
    def __init__(self,data_dir):
        self.model_name = 'Robert'
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir +'test.txt'
        self.dev_path = data_dir + 'val.txt'
        self.save_path = './save_model/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 40
        self.require_improvement = 1000
        self.num_epochs = 10
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.bert_path = './pretrained_model_weight/RoBETa/'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_class = 7
        self.class_list = ['销售介绍产品','销售介绍价格','客户咨询产品','客户咨询价格','销售介绍效果',
                           '销售介绍售后','无意图']
        self.label2id = {item:i for i,item in enumerate(self.class_list)}
        self.id2label = {i:item for i,item in enumerate(self.class_list)}


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.bert_path)
        for param in self.roberta.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size,config.num_class)

    def forward(self,x):
        context = x[0]
        mask  = x[2]
        _,pooled = self.roberta(context,attention_mask=mask)
        out = self.fc(pooled)
        return out




