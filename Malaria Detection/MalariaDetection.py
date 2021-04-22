import torch
from torch.utils.data import DataLoader,Dataset
from glob import glob
from random import shuffle,seed
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision import models
import torch.nn as nn
from torch.optim import Adam
import numpy as np
device = 'cpu'



datafolder = 'cell_images'



class CellsDataset(Dataset):
    def __init__(self,folder):
        parasaitized = glob(folder+'/Parasitized/*.png')
        uninfected = glob(folder+'/Uninfected/*.png')
        seed(10)
        self.allCells = parasaitized + uninfected
        shuffle(self.allCells)        
        self.labels = [1 if x.split('/')[1] == 'Parasitized' else 0 for x in self.allCells]

    def __len__(self):
        return len(self.allCells)

    def __getitem__(self,index):
        path = self.allCells[index]
        label = self.labels[index]
        label = torch.tensor([label]).float()
        img = cv2.imread(path)
        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = torch.tensor(img/255.0).permute(2,0,1).float()

        return img.to(device),label.to(device)

    def collateFn(self,batch):
        imgs,classes = list(zip(*batch))
        classes = torch.tensor(classes).to(device)
        classes = classes.unsqueeze(1)
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomAffine((-30,30),scale=(0.9,1.0)),
                T.ColorJitter(brightness=0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        )
        augmented = []
        for img in imgs:
            augmented.append(transform(img))
        augmented = torch.stack(augmented)

        return augmented,classes


def get_data():
    tds = CellsDataset(datafolder)
    vds = CellsDataset(datafolder)

    tdl = DataLoader(tds,32,True)
    vdl = DataLoader(vds,32,True)

    return tdl,vdl

def get_model():
    model = models.vgg16(pretrained=True,progress=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(256),
        nn.Linear(256,1)
    )

    lossFn = nn.BCEWithLogitsLoss()
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
def acc(dl):
    model.eval()
    corr=0
    samples=0

    for batch in dl:
        x,y = batch
        pred = model(x)
        for p in range(len(pred)):
            corr += (pred[p].item() > 0.5) == y[p].item()
        samples+= pred.size(0)


    print("Correct: ",corr)
    print("Samples: ",samples)
    print("Accuracy: ",corr/samples*100,"%")


totalLoss = []

for epoch in range(5):
    epochloss = []
    for xi,batch in enumerate(tdl):
        print("Epoch SIEMA: ",epoch," Batch: ",xi)
        x,y = batch
        loss = train(x,y,model,lossFn,optim)
        epochloss.append(loss)
    totalLoss.append(np.mean(epochloss))



plt.plot(totalLoss)
plt.show()

acc(tdl)
acc(vdl)


