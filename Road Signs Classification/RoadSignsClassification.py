import torch
from torch.utils.data import DataLoader,Dataset, dataloader
from glob import glob
from random import shuffle,seed
from ordered_set import OrderedSet
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.transforms import GaussianBlur
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import cv2
from torchsummary import summary
TestDatafolder = 'polishSigns/test'
TrainingDataFolder = 'polishSigns/train'
device = 'cuda'
#92 KLASY
#TODO 1- Augmentation 2- model(wlasny) 3 - model RESnet 4- testy




transform = T.Compose(
[
    T.ToPILImage(),
    T.RandomAffine(5,translate=None,scale=(0.8,1)),
    T.ColorJitter(0.05,0.03,0.03),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor()
]
)
class SignsDataSet(Dataset):
    def __init__(self,path):
        self.signs = glob(path+'/*/*.jpg')
        listofSigns = [x.split('/')[2] for x in self.signs]
        self.classesDict= dict(zip(OrderedSet(listofSigns),range(0,92)))
        print(self.classesDict)
        self.labels = [self.classesDict[x] for x in listofSigns]
        seed(10)
        toshuffle = list(zip(self.signs,self.labels))
        shuffle(toshuffle)
        self.signs,self.labels = list(zip(*toshuffle))

    def __len__(self):
        return len(self.signs)


    def __getitem__(self,index):
        filepath = self.signs[index]
        label = self.labels[index]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(64,64))
        label = torch.tensor(label).to(device)
        img = torch.tensor(img/255.0).float().to(device)
        img = img.permute(2,0,1)

        return img,label

    def collateFn(self,batch):
        imgs,clss = list(zip(*batch))
        clss = torch.tensor(clss).to(device)
        augmented = []
        for img in imgs:
            augmented.append(transform(img))
        augmented = torch.stack(augmented)
        return augmented.to(device),clss




def get_data():
    tds = SignsDataSet(TrainingDataFolder)
    tdl = DataLoader(tds,32,True,collate_fn=tds.collateFn)

    vds = SignsDataSet(TestDatafolder)
    vdl = DataLoader(vds,32,True,collate_fn=vds.collateFn)

    return tdl,vdl


def get_model():
    model = nn.Sequential(
        nn.Conv2d(3,16,3),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),

        nn.Conv2d(16,16,3),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(2),

        nn.Conv2d(16,32,3),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        nn.Conv2d(32,32,3),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        nn.Flatten(), 
        nn.Linear(128,92)

    )

    lossFn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(),lr=1e-3)


    return model.to(torch.device("cuda")),lossFn,optim


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
print(device)
summary(model,(3,64,64))


@torch.no_grad()
def acc(dl):
    model.eval()
    corr = 0
    samples = 0

    for xi,batch in enumerate(dl):
        print(xi)
        x,y = batch
        pred = model(x)
        _,val = pred.max(1)
        corr+= (val == y).sum()
        samples+=pred.size(0)

    print("Correct: ",corr)
    print("Samples: ",samples)
    print("Accuracy: ",corr/samples * 100,"%")
    return corr/samples * 100
            






totalLoss = []
for epoch in range(5):
    epochloss = []
    for xi, batch in enumerate(tdl):
        print("epoch: ",epoch," batch: ",xi)
        x,y = batch
        loss = train(x,y,model,lossFn,optim)
        epochloss.append(loss)

    totalLoss.append(np.mean(epochloss))


plt.plot(totalLoss)
plt.show()
tacc=acc(tdl)
vacc =acc(vdl)
print(tacc)
print(vacc)

model.eval()
image = cv2.imread("znaki-drogowe-wimed-1.jpg")
image = cv2.resize(image,(64,64))
image = torch.tensor(image/255.0).float().to(device)
image = image.permute(2,0,1)
image = transform(image)

# plt.imshow(image.permute(1,2,0).cpu())
# plt.show()
# image = image.permute(0,2,1)
image = image.cuda()
predicted = model(image.unsqueeze(0))
print(torch.argmax(predicted))

