#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import tensorflow as tf
# tf.keras.backend.set_floatx('float32')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Load_file import load_data,args

output_dimension=args.output_dimension
learn_rate = args.lr
num_epoches = args.epoches
batch_size=args.batch_size

data=load_data()
n=len(data)
n_train=int(n*0.8)

# 精简版
# x_train=tf.convert_to_tensor(data[:,0:-1])
# y_train=tf.convert_to_tensor(data[:,-1])
#
# tf.keras.backend.clear_session()
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(14, activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(28, activation=tf.keras.activations.relu),
#     tf.keras.layers.Dense(output_dimension)
# ])
# # model.build(input_shape=(40000,7))
# # print(model.summary())
#
# model.compile(optimizer='adam',
#               loss='mean_squared_error')
#
#
# model.fit(x_train,y_train,validation_split=0.2,batch_size=batch_size,epochs=num_epoches)
# # loss: 8.0798e-04 - val_loss: 9.1139e-04

# 标准版
train_db = tf.data.Dataset.from_tensor_slices((data[0:n_train,0:-1],data[0:n_train,-1]))
train_db = train_db.shuffle(100).batch(batch_size)
valid_db = tf.data.Dataset.from_tensor_slices((data[n_train:,0:-1],data[n_train:,-1]))
valid_db = valid_db.batch(batch_size)

class MyNeuralNet(tf.keras.Model):
    def __init__(self):
        super(MyNeuralNet,self).__init__()
        self.fc1=tf.keras.layers.Dense(14, activation=tf.keras.activations.relu)
        self.fc2=tf.keras.layers.Dense(28, activation=tf.keras.activations.relu)
        self.fc3=tf.keras.layers.Dense(output_dimension)

    def call(self, inputs):
        x=self.fc1(inputs)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

model=MyNeuralNet()
optimizer=tf.keras.optimizers.Adam(learn_rate)

train_loss_list=[]
valid_loss_list=[]

for epoch in range(num_epoches):
    train_loss_total=0.0
    valid_loss_total=0.0
    for x,y in train_db:
        # x.shape=（64,1）y.shape=（64，），out.shape=(64,1)
        # 梯度记录器，训练时需要使用它
        with tf.GradientTape() as tape:
            out = model(x)  # 通过网络获得输出
            # rf.reduce_mean计算指定轴上的平均值
            # train_loss = tf.reduce_mean(tf.keras.losses.MSE(y, out))
            train_loss = tf.keras.losses.MSE(y,tf.squeeze(out))
        # 计算梯度，并更新
        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_total += train_loss.numpy()*x.shape[0]
    train_loss_mean = train_loss_total/n_train
    train_loss_list.append(train_loss_mean)
    for x,y in valid_db:
        out=model(x)
        # valid_loss=tf.reduce_mean(tf.keras.losses.MSE(y,out))
        valid_loss = tf.keras.losses.MSE(y,tf.squeeze(out))
        valid_loss_total += valid_loss.numpy()*x.shape[0]
    valid_loss_mean = valid_loss_total/(n-n_train)
    valid_loss_list.append(valid_loss_mean)
    print(f'{epoch+1}/{num_epoches}, train_loss: {train_loss_mean:.5f}, valid_loss: {valid_loss_mean:.5f}')

