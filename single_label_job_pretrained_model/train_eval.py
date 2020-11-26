# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from single_label_job_pretrained_model.utils import get_time_dif
import pandas as pd
import os


def train(config, model, train_iter=None, dev_iter=None, test_iter=None):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    # dev_best_loss = float('inf')
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        start_time = time.time()
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            # loss = F.nll_loss(F.sigmoid(outputs),labels)
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_acc > dev_best_acc:
                    # dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        print('The time of one epoch is {}'.format(get_time_dif(start_time)))
        if flag:
            break

    predict_all = test(config, model, test_iter)
    return predict_all


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)



def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    tables = np.array([],dtype=float)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            # loss = F.nll_loss(F.sigmoid(outputs), labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()

            sigmoid_scores = F.softmax(outputs,dim=1).data.cpu().numpy()

            t_data = np.concatenate([sigmoid_scores,labels.reshape([-1,1])],axis=1)
            tables = np.append(tables,t_data)

            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)


    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        tables = np.reshape(tables, [-1, 4])
        df_t = pd.DataFrame(columns=['有意图', '无意图', '其他', '真实值'], data=tables)
        for name in config.names:
            if os.path.exists('./sigmoid_{}.csv'.format(name)):
                continue
            df_t.to_csv('./sigmoid_{}.csv'.format(name))
            break
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def predict(config,model,test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    predict_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts in test_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    return predict_all

