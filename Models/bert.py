from transformers import BertModel,BertTokenizer
import torch
from torch import nn
class Config(object):
    def __init__(self,data_dir):
        self.model_name = 'bert'
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir +'test.txt'
        self.dev_path = data_dir + 'dev.txt'
        self.save_path = './save_model/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 512
        self.require_improvement = 1000
        self.num_epochs = 10
        self.batch_size = 16
        self.learning_rate = 3e-5
        self.bert_path = './pretrained_model_weight/bert/'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.class_list = ['0','1','2','3','4','5','6','7','8','9']
        self.num_class = len(self.class_list)




class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size,config.num_class)

    def forward(self,x):
        context = x[0]
        mask  = x[2]
        print(context.size())
        _,pooled = self.bert(context,attention_mask=mask)
        out = self.fc(pooled)
        return out
