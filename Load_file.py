#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import argparse
import pandas as pd
import numpy as np

# 读取数据
def load_data():
    data=pd.read_csv("crackdata.csv")
    prepareddata=data['wmax']
    data=data.drop(['wmax'],axis=1).drop(['Unnamed: 0'],axis=1)
    data=data.apply(lambda x :(x-x.mean())/(x.std())) #零-均值标准化
    data['wmax']=prepareddata
    return np.array(data)

# 设定参数
parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--output-dimension', type=int, default=1, metavar='N',
                    help='output dimension (default: 1)')
parser.add_argument('--epoches', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
args = parser.parse_args()
