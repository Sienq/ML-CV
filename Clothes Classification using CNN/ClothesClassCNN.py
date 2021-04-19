import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import  datasets
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import ToPILImage, ToTensor
device = 'cpu'


class ClothesDataSet(Dataset):
    def __init__(self,x,y,aug=0):
        x = x.float()
        self.X = x.view(-1,1,28,28)/255.0
        self.Y = y
        self.aug = aug
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X
        y = self.Y
        return x[index].float().to(device),y[index].to(device)


def get_data():
    TRdatafolder = 'data/training'
    VALdatafolder = 'data/validation'

    trFmnist = datasets.FashionMNIST(TRdatafolder,True,download=True)
    valFmnist = datasets.FashionMNIST(VALdatafolder,False,download=True)

    trImg = trFmnist.data
    trTar = trFmnist.targets

    valImg = valFmnist.data
    valTr = valFmnist.targets

    tds = ClothesDataSet(trImg,trTar,0)
    vds = ClothesDataSet(valImg,valTr,0)

    tdl = DataLoader(tds,32,True)
    vdl = DataLoader(vds,32,True)

    return tdl,vdl


def get_model():
    model = nn.Sequential(
        nn.Conv2d(1,64,(3,3)),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.BatchNorm2d(64),
        nn.MaxPool2d((2,2),2),

        nn.Conv2d(64,128,(3,3)),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.BatchNorm2d(128),
        nn.MaxPool2d((2,2),2),

        nn.Flatten(),
        nn.Linear(3200,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(128),
        nn.Linear(128,10)
    )

    lossFn = nn.CrossEntropyLoss()

    optim = Adam(model.parameters(),lr=1e-3)

    return model,lossFn,optim

def train(x,y,model,lossFn,optim):
    model.train()
    pred = model(x)
    loss = lossFn(pred,y)
    loss.backward()
    optim.step()
    optim.zero_grad()

    return loss.item()

model,lossFn,optim = get_model()
tdl,vdl = get_data()

@torch.no_grad()
def acc(loader):
    model.eval()
    corr = 0
    samples = 0
    for x,y in loader:
        x.to(device)
        y.to(device)
        pred = model(x)
        _,val = pred.max(1)
        corr+= (val == y).sum()
        samples+=pred.size(0)

    print("Correct: ",corr)
    print("Samples: ",samples)
    print("Accuracy: ",corr/samples * 100,"%")

lossOverTraining = []
for epoch in range(15):
    epochloss = []
    for xi, batch in enumerate(tdl):
        print('epoch: ',epoch,' batch: ',xi)
        x,y = batch
        loss = train(x,y,model,lossFn,optim)
        epochloss.append(loss)
    lossOverTraining.append(np.mean(epochloss))

print("TRAINING")
acc(tdl)
print("TEST")
acc(vdl)

summary(model,(1,28,28))
plt.plot(lossOverTraining)
plt.show()
