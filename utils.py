# -*- coding: utf-8 -*-
# @Time : 2021/5/31 下午10:12
# @Author : ddz

import pandas as pd
import numpy as np
import random
import torch

from sklearn.metrics import confusion_matrix


def seed_everything(seed=2019):
    '''
    设置随机种子，最好在训练的时候调用
    '''
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    '''
    获取机器的cpu或者gpu
    '''
    device_all = []
    n_gpu = torch.cuda.device_count()
    for index in range(n_gpu):
        device_all.append(torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu"))

    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
        return torch.device("cpu")
    return device_all, n_gpu


def seq_statistics(tokens):
    '''
    统计tokens长度，获取最优的 seq
    '''
    # 分词
    #     tokens = nltk.word_tokenize(sentence)

    count_num_dict = {}

    length = len(tokens)
    print(length)
    if len(str(length)) == 1:  # 1 位数
        key = '10'
    elif len(str(length)) == 2:  # 两位数
        key = str(str(length)[0] + '0')
    elif len(str(length)) == 3:  # 三位数
        key = str(str(length)[0:2] + '0')
    else:
        print('tokens is so long')
        key = '>1000'

    if key in count_num_dict.keys():
        count_num_dict[key] += 1
    else:
        count_num_dict[key] = 1

    return count_num_dict


def calFPRAndFOR(y_label, y_pred, category=2):
    '''计算混淆矩阵，误报率和漏报率

    参数含义：
    y_label -- 实际标签
    y_pred -- 模型预测结果
    category -- 分类类型
    '''
    # 计算三分类混淆矩阵，误报率和漏报率
    if category == 3:
        confusionMatrix = confusion_matrix(y_label, y_pred)
        print('混淆矩阵：')
        print(confusionMatrix)
        m = {0: '色情', 1: '勒索', 2: '拦截马'}
        allSum = sum(sum(confusionMatrix))
        n = len(confusionMatrix)
        for i in range(n):
            rowsum, colsum = sum(confusionMatrix[i]), sum(confusionMatrix[r][i] for r in range(n))
            try:
                if rowsum == 0:
                    print('%s类：误报率: %.3f%%，漏报率: %.3f%%' %
                          (m[i], ((colsum - confusionMatrix[i][i]) / (allSum - rowsum)) * 100, 0))
                else:
                    print('%s类：误报率: %.3f%%，漏报率: %.3f%%' %
                          (m[i], ((colsum - confusionMatrix[i][i]) / (allSum - rowsum)) * 100,
                           ((rowsum - confusionMatrix[i][i]) / float(rowsum)) * 100))
            except ZeroDivisionError:
                print('precision: %s' % 0, 'recall: %s' % 0)

    if category == 2:
        # 计算漏报率和误报率
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
        acc = int(tp + tn) / int(tp + fn + tn + fp)
        print('准确率为：%.2f5%%' % (acc * 100))
        try:
            fnr = int(fn) / int(tp + fn)
            print('漏报率：%.2f%%' % (fnr * 100))
        except ZeroDivisionError:
            print('该数据集中无"黑样本"，无法计算漏报！')
        try:
            fpr = int(fp) / int(tn + fp)
            print('误报率：%.2f%%' % (fpr * 100))
        except ZeroDivisionError:
            print('该数据集中无"白样本"，无法计算误报！')


def regularization_loss(model, weight_decay):
    l2_loss = torch.tensor(0.)
    for param in model.parameters():
        l2_loss += torch.norm(param)
    return weight_decay * l2_loss.item()
