import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torchvision import transforms as T
device = 'cpu'


inputs = torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[8.,9.],[10.,11.],[12.,13.],[14.,15.],[16.,17.],[18.,19.],[20.,21.]]).float().to(device)
targets = torch.tensor([[3.],[7.],[11.],[15.],[17.],[21.],[25.],[29.],[33.],[37.],[41.]]).float().to(device)
transform = T.Compose(
    T.Normalize([[0.485, 0.456, 0.406]],[0.229, 0.224, 0.225])
)


print(inputs.max())

class NumbersDataset(Dataset):
    def __init__(self,x,y,transform = None):
        self.X = x
        self.Y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        return x.to(device),y.to(device)


def get_data():
    tds = NumbersDataset(inputs,targets,transform)
    tdl = DataLoader(tds,1,shuffle=True)

    return tdl

def get_model():
    model = nn.Sequential(
        nn.Linear(2,8),
        nn.ReLU(),
        nn.Linear(8,1),
    )

    loss = nn.MSELoss()

    optim = SGD(model.parameters(),lr=1e-4)

    return model.to(device),loss,optim

def train(x,y,model,lossFn,optim):
    model.train()
    pred = model(x)
    loss = lossFn(pred,y)
    loss.backward()
    optim.step()
    optim.zero_grad()
    return loss.item()



model,lossFn,optim = get_model()
tdl = get_data()
losstab = []
for epoch in range(30):
    print(epoch)
    for xi ,batch in enumerate(iter(tdl)):
        x,y = batch
        loss = train(x,y,model,lossFn,optim)
        losstab.append(loss)


z = torch.tensor([25,26]).float()

plt.plot(losstab)
plt.show()
predicted = model(z)


print(predicted)