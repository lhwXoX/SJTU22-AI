import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.densenet import DenseNet

from models.counting import CountingDecoder as counting_decoder
from counting_utils import gen_counting_label


class CAN(nn.Module):
    def __init__(self, params=None):
        super(CAN, self).__init__()
        self.params = params
        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)
        self.in_channel = params['counting_decoder']['in_channel']
        self.out_channel = params['counting_decoder']['out_channel']
        self.counting_decoder1 = counting_decoder(self.in_channel, self.out_channel, 3)
        self.counting_decoder2 = counting_decoder(self.in_channel, self.out_channel, 5)
        #新增了两个counting_decoder,对应两个7*7和9*9的卷积
        self.counting_decoder3 = counting_decoder(self.in_channel, self.out_channel, 7)
        self.counting_decoder4 = counting_decoder(self.in_channel, self.out_channel, 9)

        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()
        self.counting_loss = nn.SmoothL1Loss(reduction='mean')

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        #定义可训练权重6个
        self.weight1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        self.weight3 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight4 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.weight5 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.weight6 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)
        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        counting_labels = gen_counting_label(labels, self.out_channel, True)

        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        #用新增的counting_decoder得到结果
        counting_preds3, _ = self.counting_decoder3(cnn_features, counting_mask)
        counting_preds4, _ = self.counting_decoder4(cnn_features, counting_mask)
        #确保权重之和为1
        weight_sum_1 = self.weight1 + self.weight2
        weight_sum_2 = self.weight3 + self.weight4
        weight_sum_3 = self.weight5 + self.weight6

        normalized_weight1 = self.weight1 / weight_sum_1
        normalized_weight2 = self.weight2 / weight_sum_1
        normalized_weight3 = self.weight3 / weight_sum_2
        normalized_weight4 = self.weight4 / weight_sum_2
        normalized_weight5 = self.weight5 / weight_sum_3
        normalized_weight6 = self.weight6 / weight_sum_3
        '''
        #原文使用的average
        counting_preds = (counting_preds1 + counting_preds2) / 2
        counting_loss = self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) \
                        + self.counting_loss(counting_preds, counting_labels)
        '''
        #得到两个counting vectors
        counting_preds_1 = normalized_weight1 * counting_preds1 + normalized_weight2 * counting_preds2
        counting_preds_2 = normalized_weight3 * counting_preds3 + normalized_weight4 * counting_preds4

        #计算两个counting vectors一起的counting loss
        counting_loss = (self.counting_loss(counting_preds_1, counting_labels) + self.counting_loss(counting_preds_2, counting_labels) + self.counting_loss(counting_preds1, counting_labels) + self.counting_loss(counting_preds2, counting_labels) + self.counting_loss(counting_preds3, counting_labels) + self.counting_loss(counting_preds4, counting_labels))/2
        
        word_probs, word_alphas = self.decoder(cnn_features, labels, counting_preds_1, counting_preds_2, images_mask, labels_mask, is_train=is_train)
        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss
        return word_probs, counting_preds_1, counting_preds_2, word_average_loss, counting_loss
