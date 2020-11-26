import logging
import os
import math
from transformers import BertTokenizer,RobertaModel,LongformerModel,BertModel
from transformers.modeling_longformer import LongformerSelfAttention
import torch
from torch import nn
class Config(object):
    def __init__(self,data_dir):
        self.model_name = 'longformer'
        self.train_path = data_dir + 'train.txt'
        self.test_path = data_dir +'test.txt'
        self.dev_path = data_dir + 'dev.txt'
        self.save_path = './save_model/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 3072
        self.require_improvement = 1000
        self.num_epochs = 1
        self.batch_size = 7
        self.learning_rate = 5e-6
        self.bert_path = './pretrained_model_weight/bert/'
        self.longformer_path = './pretrained_model_weight/longformer/'
        self.tokenizer = BertTokenizer.from_pretrained(self.longformer_path)
        self.hidden_size = 768
        self.names = ['test','dev']

        self.class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.num_class = len(self.class_list)


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        # create_long_model(config,32,3072+2)
        print('转换模型完毕')
        self.longformer = LongformerModel.from_pretrained(config.longformer_path)
        for param in self.longformer.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size,config.num_class)

    def forward(self,x):
        context = x[0]
        mask  = x[2]
        _,pooled = self.longformer(context,attention_mask=mask)
        out = self.fc(pooled)
        return out

def create_long_model(config,attention_window,max_pos):
    my_config = config
    model = RobertaModel.from_pretrained(my_config.bert_path)

    config = model.config
    # extend position embeddings

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[:]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn
    model.save_pretrained(my_config.longformer_path)





