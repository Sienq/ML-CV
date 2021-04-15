import torch
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
import  matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torchsummary import summary

device = 'cpu'
dataTR = 'data/training'
dataVAL = 'data/validation'

fmnistTR =  datasets.FashionMNIST(dataTR,True,download=True)
trData = fmnistTR.data
trTargets = fmnistTR.targets
fmnistVAL = datasets.FashionMNIST(dataVAL,False,download=True)

valData = fmnistVAL.data
valTargets = fmnistVAL.targets


class ClothesDataset(Dataset):
    def __init__(self,x,y):
        x = x.float()/255.0
        self.X = x.view(-1,28*28).float()
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]


        return x.to(device),y.to(device)



def get_data():
    tds = ClothesDataset(trData,trTargets)
    trainingDataLoader = DataLoader(tds,32,True)
    vds = ClothesDataset(valData,valTargets)
    validationDataLOader = DataLoader(vds,32,True)

    return trainingDataLoader,validationDataLOader

def get_model():
    model = nn.Sequential(
        nn.Linear(28*28,400),
        nn.BatchNorm1d(400),
        nn.ReLU(),
        nn.Linear(400,10)
    ).to(device)

    lossFn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(),lr=1e-4)

    return model,lossFn,optim


def train(x,y,model,lossFn,opt):
    model.train()
    pred = model(x)
    loss = lossFn(pred,y)
    loss.backward()
    opt.step()
    opt.zero_grad()

    return loss.item()

tdl,vdl = get_data()
model,lossFn,optim = get_model()


@torch.no_grad()
def acc():
    model.eval()
    corr = 0
    samples = 0

    for x,y in vdl:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _,predictions = scores.max(1)
        corr += (predictions==y).sum()
        samples += predictions.size(0)

    print('Correct: ',corr)
    print('Samples: ',samples)
    print('Accuracy: ',(corr/samples)*100,'%')







totalLoss = []

summary(model,torch.zeros(1,28*28))
for epoch in range(25):
    epochloss = []
    print(epoch)
    for xi,batch in enumerate(tdl):
        # print("epoch: ",epoch," batch: ",xi)
        x,y = batch
        batchLoss = train(x,y,model,lossFn,optim)
        epochloss.append(batchLoss)
    totalLoss.append(np.mean(epochloss))



plt.plot(totalLoss)
plt.show()

acc()

