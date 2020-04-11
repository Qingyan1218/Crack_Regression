#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Jerry Zhu

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from Load_file import load_data,args

output_dimension=args.output_dimension
learn_rate = args.lr
num_epoches = args.epoches
batch_size=args.batch_size

data=load_data()
training_data_list=data[:int(0.8*len(data))]
valid_data_list=data[int(0.8*len(data)):]
train_data=torch.from_numpy(training_data_list).type(torch.float32)
valid_data=torch.from_numpy(valid_data_list).type(torch.float32)

# 定义数据加载方式
class MyDataset():
    def __init__(self, data):
        self.size = data.shape[0]
        self.data=data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        arg=self.data[idx][0:-1]
        label=self.data[idx][-1]
        sample = {'arg': arg, 'label': label}
        return sample

# 加载数据
train_dataset = MyDataset(train_data)
valid_dataset = MyDataset(valid_data)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
data_loaders = {'train': train_dataloader, 'val': test_dataloader}

# 定义模型
class MyNeuralNet(torch.nn.Module):
    def __init__(self):
        super(MyNeuralNet,self).__init__()
        self.FullConnection = torch.nn.Sequential(
            torch.nn.Linear(7, 14),
            torch.nn.ReLU(),
            torch.nn.Linear(14, 28),
            torch.nn.ReLU(),
            torch.nn.Linear(28, output_dimension)
        )

    def forward(self, x):
        x=self.FullConnection(x)
        return x

# 定义模型训练函数
def train(model, criterion, optimizer, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for idx,data in enumerate(data_loaders[phase]):
                inputs = data['arg']
                labels_true = data['label']
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x = torch.squeeze(model(inputs))
                    loss = criterion(x, labels_true)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 因为loss已经是平均值了，所以要乘以batch_size，因为每个batch_size会不一样
                running_loss += loss.item() * inputs.size(0)

            # 最后除以总数变成损失的均值
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    return Loss_list

# 定义损失和优化器
model=MyNeuralNet()
criterion=torch.nn.MSELoss()
# optimizer = optim.SGD(model.parameters(),lr=learn_rate,momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=learn_rate)

# 训练
Loss_list = train(model, criterion,optimizer, num_epochs=num_epoches)
# train Loss: 0.0007 val Loss: 0.0007

