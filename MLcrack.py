import pandas
import numpy
import time
data=pandas.read_csv("crackdata.csv")

#将数据的租后一列根据裂缝值划分成两种类型，用0和1代表
prepareddata=data['wmax'].apply(lambda x: 1 if x<=0.2 else 0)
data=data.drop(['wmax'],axis=1).drop(['Unnamed: 0'],axis=1)

data['OK']=prepareddata

#模型的80%用于学习，剩下的用于测试
data_train=data[:int(0.8*len(data))]
data_test=data[int(0.8*len(data)):]

collen=len(data.columns)-1
collist=list(range(collen))
crack_feats=data_train.iloc[:,collist]
crack_target=data_train["OK"]
crack_test=data_test.iloc[:,collist]

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
modellist=[LogisticRegression(solver='liblinear'),SVC(gamma=0.1)]

def train(modelname,features,target):
    model=modelname
    model.fit(features,target)
    return model

def predict(model,new_features):
    preds=model.predict(new_features)
    return preds

for model in modellist:
    startPerfCounter = time.perf_counter()
    modeltrained=train(model,crack_feats,crack_target)
    predictions=predict(modeltrained,crack_test)
    print(model.score(crack_feats,crack_target))
    endPerfCounter = time.perf_counter()
    totaltime = endPerfCounter - startPerfCounter
    print('time is uesd:', totaltime)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(crack_feats)
print(pca.components_)
print(pca.explained_variance_)


