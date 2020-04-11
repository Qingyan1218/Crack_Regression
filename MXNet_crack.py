#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import mxnet
from mxnet import nd, autograd
from Load_file import load_data,args

output_dimension=args.output_dimension
learn_rate = args.lr
num_epoches = args.epoches
batch_size=args.batch_size

data=load_data()
n=len(data)
n_train=int(n*0.8)
train_features=nd.array(data[0:n_train,0:-1])
train_labels=nd.array(data[0:n_train,-1])

valid_features=nd.array(data[n_train:,0:-1])
valid_labels=nd.array(data[n_train:,-1])

train_set=mxnet.gluon.data.ArrayDataset(train_features,train_labels)
train_iter=mxnet.gluon.data.DataLoader(train_set,batch_size,shuffle=True)
valid_set=mxnet.gluon.data.ArrayDataset(valid_features,valid_labels)
valid_iter=mxnet.gluon.data.DataLoader(valid_set,batch_size)

model = mxnet.gluon.nn.Sequential()
model.add(mxnet.gluon.nn.Dense(14,activation='relu'),
          mxnet.gluon.nn.Dense(28,activation='relu'),
          mxnet.gluon.nn.Dense(output_dimension))

model.initialize(mxnet.init.Normal(sigma=0.01))

loss=mxnet.gluon.loss.L2Loss()
# def mse(y_hat,y):
#     return (y_hat-y)**2
# loss=mse

trainer=mxnet.gluon.Trainer(model.collect_params(),'Adam',{'learning_rate':learn_rate})

train_loss_list=[]
valid_loss_list=[]
for epoch in range(num_epoches):
    train_loss_total=0.0
    valid_loss_total=0.0
    for X,y in train_iter:
        # X.shape=(64,7),y.shape=(64,),model(X).shape=(64,1)
        with autograd.record():
            train_loss=loss(nd.squeeze(model(X)),y)
            # train_loss.shape=(64,)
        train_loss.backward()
        trainer.step(batch_size)
        # mxnet的loss输出是0.5*(out-y)^2的一个向量，因此求和之后乘以2才是标准mse
        train_loss_total += train_loss.asnumpy().sum()
    train_loss_mean = train_loss_total/n_train
    train_loss_list.append(train_loss_mean)
    for X,y in valid_iter:
        valid_loss=loss(nd.squeeze(model(X)),y)
        valid_loss_total += valid_loss.asnumpy().sum()
    valid_loss_mean = valid_loss_total/(n-n_train)
    valid_loss_list.append(valid_loss_mean)
    print(f'{epoch+1}/{num_epoches}, train_loss: {train_loss_mean:.5f}, valid_loss: {valid_loss_mean:.5f}')

# 50/50, train_loss: 0.00418, valid_loss: 0.00416
# 100/100, train_loss: 0.00086, valid_loss: 0.00116
# 可能是mxnet的loss比较小，因此反传的loss也比较小，所以需要更多的epoches才能达到同样的loss
