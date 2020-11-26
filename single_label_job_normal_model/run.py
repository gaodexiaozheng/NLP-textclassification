import torch
import numpy as np
from importlib import import_module
from single_label_job_normal_model.train_eval import train,init_network,predict
from single_label_job_normal_model.utils import build_dataset,build_iterator,build_predict_dataset,pred_iterator
import argparse
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                        help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
args = parser.parse_args()
if __name__ == '__main__':

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    x = import_module('Models.' + model_name)
    config = x.Config('../data/THUCNews/')
    flags = 'normal'
    if model_name == 'FastText':
        config.n_gram_vocab = config.n_vocab_c * 2
        flags = 'fasttext'


    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    model = x.Model(config).to(config.device)
    print(model.parameters)

    train_set,dev_set,test_set = build_dataset(config)
    train_iter = build_iterator(train_set, config,flags)
    dev_iter = build_iterator(dev_set, config,flags)
    test_iter = build_iterator(test_set,config,flags)


    if model_name == 'Transformer':
        init_network(model)

    train(config, model, train_iter, dev_iter, test_iter)

    # Below code is just for prediction

    # predict_data, pred_pd = build_predict_dataset(config)
    # pred_iter = pred_iterator(predict_data, config)
    # result = predict(config, model, pred_iter)
    # result = result.astype(int).tolist()
    # pred_pd['model_result'] = result
    # pred_pd.to_csv('data_pred.txt', index=False, encoding='utf_8_sig')











