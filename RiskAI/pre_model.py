import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
def returnModel(pretrain):
    if pretrain == 'y':
        model = models.resnet18(pretrained = True)
    elif pretrain == 'n':
        model = models.resnet18(pretrained = False)
    #print(model)
    model.fc = Identity()
    #print(model)
    #model.train()
    # for param in model.parameters():
    #     param.requires_grad = False
    return model

def returnModel2(pretrain):
    if pretrain == 'y':
        model = models.googlenet(pretrained = True)
    elif pretrain == 'n':
        model = models.googlenet(pretrained = False)
    #print(model)
    model.fc = Identity()
    #print(model)
    #model.train()
    # for param in model.parameters():
    #     param.requires_grad = False
    return model


def returnModel3(pretrain):
    if pretrain == 'y':
        model = models.mobilenet_v3_large(pretrained = True)
    elif pretrain == 'n':
        model = models.mobilenet_v3_large(pretrained = False)
    #print(model)
    model.classifier = Identity()
    #print(model)
    #model.train()
    # for param in model.parameters():
    #     param.requires_grad = False
    return model

def returnModel4(pretrain):
    if pretrain == 'y':
        model = models.shufflenet_v2_x1_0(pretrained = True)
    elif pretrain == 'n':
        model = models.shufflenet_v2_x1_0(pretrained = False)
    model.fc = Identity()
    #print(model)
    #model.train()
    # for param in model.parameters():
    #     param.requires_grad = False
    return model

class Pretr_Att(nn.Module):
    def __init__(self, pretrain, model = 'res', attention_flag = True):
        super(Pretr_Att, self).__init__()
        #self.L = 512
        #self.L = 960
        #self.L = 1024
        self.D = 128
        self.K = 1
        self.attention_flag = attention_flag
        if model == 'res':
            self.L = 512
            self.extractor = returnModel(pretrain)
            print('Loaded: Resnet18')
        elif model == 'mob':
            self.L = 960
            self.extractor = returnModel3(pretrain)
            print('Loaded: MobileNet')
        elif model == 'goo':
            self.L = 1024
            self.extractor = returnModel2(pretrain)
            print('Loaded: GoogleNet')
        elif model == 'shu':
            self.L = 1024
            self.extractor = returnModel4(pretrain)
            print('Loaded: ShuffleNet')
        else:
            self.L = 512
            self.extractor = returnModel(pretrain)
            print('Loaded: Resnet18')
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        #print(x.shape)
        H = self.extractor(x)
        if self.attention_flag:
            A_V = self.attention_V(H)  # NxD
            A_U = self.attention_U(H)  # NxD
            A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
    
            M = torch.mm(A, H)  # KxL
        else:
            M = torch.mean(H,dim=0)
            A = torch.tensor(1)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A



    def calculate_objective(self, X, Y, weighted = False):
        Y = Y.float()
        Y_prob, Y_hat, A = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        if weighted:
            neg_log_likelihood = -1. * (0.75 * Y * torch.log(Y_prob) + 0.25 * (1. - Y) * torch.log(1. - Y_prob))
        else: 
            neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        #neg_log_likelihood = -1. * (0.75 * ((1-Y_prob)**2) * Y * torch.log(Y_prob) + 0.25 * (1. - Y) * (Y_prob**2) * torch.log(1. - Y_prob))

        return error, Y_hat, Y_prob, neg_log_likelihood
    
    