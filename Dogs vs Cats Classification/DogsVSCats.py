import torch
from torch.nn.modules.flatten import Flatten
from torch.utils.data import  DataLoader,Dataset
from glob import glob
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as T
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam
import numpy as np
from random import shuffle
 #1000 = 70% 8000 = 85%
#TODO sprawdzic 8k bez aug
device = 'cpu'
TRfolder = "D&C/training_set/training_set/"
TEfolder = "D&C/test_set/test_set/"
class DCdataset(Dataset):
    def __init__(self,path,aug = False):
        cats = glob(path+"cats/*.jpg")
        dogs = glob(path+"dogs/*.jpg")
        self.imgsPath = cats[:4000] + dogs[:4000]
        shuffle(self.imgsPath)
        self.labels = [0 if pet.split('/')[-1].startswith('dog') else 1 for pet in self.imgsPath]
        self.aug = aug

    def __len__(self):
        return len(self.imgsPath)

    def __getitem__(self,index):
        path = self.imgsPath[index]
        label = self.labels[index]
        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = torch.tensor(img/255.0).float()
        img = img.permute(2,0,1)

        return img.to(device),torch.tensor([label]).to(device).float()

    def collateFn(self,batch): 
        imgs, clss = list(zip(*batch))
        clss = torch.tensor(clss)
        clss= clss.unsqueeze(1)
        if self.aug:
            transform = T.Compose(
                [
                    T.ToPILImage(),
                    T.RandomAffine((-10,10),(0.1,0.2),scale=(0.9,1)),
                    T.ColorJitter(brightness=0.1,contrast=0.1),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ]
            )
            augmented = []
            for img in imgs:
                augmented.append(transform(img))
            
            augmented = torch.stack(augmented)

            return augmented.to(device),clss.to(device)
        else:
            return imgs.to(device),clss.to(device)


def get_data():
    tds = DCdataset(TRfolder,aug=True)
    vds = DCdataset(TEfolder,aug=True)

    tdl = DataLoader(tds,32,True,drop_last=True,collate_fn=tds.collateFn)
    vdl = DataLoader(vds,32,shuffle=True,drop_last=True,collate_fn=vds.collateFn) 

    return tdl,vdl

def get_model():
    model = nn.Sequential(
        nn.Conv2d(3,64,3),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),

        nn.Conv2d(64,512,3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2),

        nn.Conv2d(512,512,3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2),

        nn.Conv2d(512,512,3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2),

        nn.Conv2d(512,512,3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2),


        nn.Conv2d(512,512,3),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(512,1),
        nn.Sigmoid()
    )

    lossFn = nn.BCELoss()
    optim = Adam(model.parameters(),lr=1e-3)
    return model.to(device),lossFn,optim


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

summary(model,(3,224,224))

@torch.no_grad()
def acc(dl):
    model.eval()
    corr = 0
    sample = 0
    for batch in dl:
        x,y = batch
        pred = model(x)
        for pr in range(len(pred)):
            corr+= (pred[pr].item() > 0.5) == y[pr].item()    
        sample+= pred.size(0)
    print("Correct: ",corr)
    print("Samples: ",sample)
    print("Accuracy: ",corr/sample * 100,'%')

totalloss = []
for epoch in range(5):
    epochloss = []
    for ex,batch in enumerate(tdl):
        print("Epoch: ",epoch," Batch: ",ex)
        x,y = batch
        loss = train(x,y,model.cpu(),lossFn,optim)
        epochloss.append(loss)
    totalloss.append(np.mean(epochloss))
    epochloss = []

acc(tdl)
acc(vdl)
plt.plot(totalloss)
plt.show()
