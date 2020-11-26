# coding: UTF-8
import time
import torch
import numpy as np
from single_label_job_pretrained_model.train_eval import train,predict
from importlib import import_module
import argparse
from single_label_job_pretrained_model.utils import build_dataset, build_iterator,build_predict_dataset,pred_iterator

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, roberta, albert, longformer')
args = parser.parse_args()

if __name__ == '__main__':
    data_dir = '../data/THUCNews/'
    model_name = args.model
    x = import_module('models.'+model_name)
    config = x.Config(data_dir)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    model = x.Model(config).to(config.device)

    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    train(config,model,train_iter,dev_iter,test_iter)

    # predict_data, pred_pd = build_predict_dataset(config)
    # pred_iter = pred_iterator(predict_data, config)
    #
    # result = predict(model,pred_iter)
    #
    # pred_pd['model_result'] = result
    # pred_pd.to_csv('data_pred.txt',index=False)



