from torch.nn import functional as F
import torch
import numpy as np

class single_loss_fun(object):

    def __init__(self,config,name):
        self.class_num = config.class_num
        self.name = name

    def loss_fn(self):
        if self.name == 'crossentropy':
            return F.cross_entropy
        elif self.name == 'f1loss':
            return self.f1_loss
        elif self.name == 'priorloss':
            return self.categorical_crossentropy_with_prior
        elif self.name == 'diceloss':
            return self.dice_loss
        elif self.name == 'focalloss':
            return self.focal_loss


    def one_hot_label(self,y_true):
        batch_size = y_true.size(0)
        y_one_hot = torch.zeros(batch_size, self.class_num).scatter_(1, y_true, 1)
        return y_one_hot

    def f1_loss(self,y_pre, y_true, epsilon=10 - 6):
        y_true = self.one_hot_label(y_true)
        tp = torch.sum(y_true * y_pre, dim=0)
        fp = torch.sum((1 - y_true) * y_pre, dim=0)
        fn = torch.sum(y_true * (1 - y_pre), dim=0)

        p = tp / (tp + fp + epsilon)
        r = tp / (tp + fn + epsilon)

        f1 = 2 * p * r / (p + r + epsilon)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1 - torch.mean(f1)

    def focal_loss(self,y_pre,y_true):
        alpha = 0.25
        gamma = 2
        eps = 1e-6
        y_true = self.one_hot_label(y_true)
        input_soft = F.softmax(y_pre, dim=1) + eps
        weight = torch.pow(1. - input_soft, gamma)
        focal = -alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(y_true * focal, dim=1)
        return loss_tmp.mean()

    def categorical_crossentropy_with_prior(self,y_pred,y_true, tau=1.0):
        y_true = self.one_hot_label(y_true)
        prior = np.array([1/10 for i in range(10)])
        log_prior = torch.cuda.FloatTensor(np.log(prior + 1e-8))
        y_pred = y_pred + tau * log_prior
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def dice_loss(self,y_pred, y_true):

        def dice(y_pred, y_true):
            smooth = 1

            intersection = y_pred * y_true
            loss = 2 * (intersection.sum(-1) + smooth) / (y_pred.sum(-1) + y_true.sum(-1) + smooth)
            loss = 1 - loss.mean()
            return loss

        y_true = self.one_hot_label(y_true)
        C = y_true.size(1)
        total_loss = 0
        for i in range(C):
            loss = dice(y_pred[:, i], y_true[:, i])
            total_loss += loss
        return total_loss


