import numpy
import pandas as pd
from neuralnet import *
import time

startPerfCounter=time.perf_counter()

# number of input, hidden and output nodes
input_nodes=7
hidden_nodes=100
output_nodes=1

# learning rate is 0.2, it is efficient
learning_rate=0.2

# create instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the training data csv file into a list
data=pd.read_csv("crackdata.csv")
prepareddata=data['wmax']
data=data.drop(['wmax'],axis=1).drop(['Unnamed: 0'],axis=1)
data=(data - data.min())/(data.max() - data.min()) #最小-最大规范化
#data=(data - data.mean())/data.std() #零-均值规范化
data['wmax']=prepareddata
training_data_list=numpy.array(data[:int(0.8*len(data))])
test_data_list=numpy.array(data[int(0.8*len(data)):])
# train the neural network
# epochs is the nunber of times the training data set is used for training
epochs=5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        inputs=record[0:-1]
        targets=record[-1]
        n.train(inputs,targets)

# test the neural network
number=0
for record in test_data_list:
    inputs=record[0:-1]
    correct_label=record[-1]
    outputs=n.query(inputs)
    delta=correct_label-outputs[0][0]
    if delta<0.005:
        number +=1

score=number/len(test_data_list)
print('the score is %s' % score)

#scorecard_array=numpy.asarray(scorecard)
#succeedrate=scorecard_array.sum()/scorecard_array.size
#print(succeedrate)
endPerfCounter=time.perf_counter()
totaltime=endPerfCounter-startPerfCounter
print('time is uesd:',totaltime)








