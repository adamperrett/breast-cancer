import torch
import torch.nn as nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
    
def returnModel(pretrain, replicate):
    model = models.resnet50(pretrained = pretrain)
    model.fc = Identity()
    if not replicate:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


class Pvas_Model(nn.Module):
    def __init__(self, pretrain, replicate):
        super(Pvas_Model, self).__init__()
        
        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate)
        self.D = 2048 ##out of the model
        self.K = 1024 ##intermidiate
        self.L = 512  
        
        self.regressor = nn.Sequential(
            nn.Linear(self.D, self.K),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.K, self.L),
            nn.Dropout(p=0.5),
            nn.ReLU(),  
            nn.Linear(self.L, 1)
            )
        
        
    def forward(self, x):
        H = self.extractor(x)
        r = self.regressor(H)
        return r
        
    
    def objective(self, X, Y):
        Y = Y.unsqueeze(1)
        R = self.forward(X)
        loss = nn.MSELoss()
        return loss(R, Y), R
        
        
        
        
        
        