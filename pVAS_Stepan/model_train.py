import argparse
import os
import time

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable


#from pvas_loader import ProcasLoader
from pVAS_Stepan.pvas_loader_proc import ProcasLoader
from pVAS_Stepan.model import Pvas_Model
from pVAS_Stepan.pvas_utils import EarlyStopper


def get_args():
    parser = argparse.ArgumentParser(description='Training pVAS sr')
    parser.add_argument('--csf', action='store_true', default=False, help='Are we training on the csf or not?')
    parser.add_argument('--no-cuda', action='store_true', default = False, help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=2, metavar='N')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR')
    parser.add_argument('--bs', type=int, default=1, metavar='B')
    parser.add_argument('--view', type=str, default='MLO', help = 'Which mammographic view to use')
    parser.add_argument('--pretrained', action='store_true', default=False, help = 'Use pretrained network or not?')
    parser.add_argument('--replicate', action='store_true', default=True, help = 'Replicate or not? IF no will replace the first CNN layer.')
    parser.add_argument('--priors', action='store_true', default=False, help = 'Test on the priors automatically on the best model?')
    parser.add_argument('--name', type=str, default = 'model', help = 'What to call the model test?')
    return parser.parse_args()

def train(epoch):
    model.train()
    train_loss = 0
    total_iterations = len(train_loader.dataset) // train_loader.batch_size
    print(f'Begin training epoch {epoch}: ')
    with tqdm(total=total_iterations, unit="batch") as t:
        for batch_idx, (image, label, name) in enumerate(train_loader):
            if args.cuda:
                image, label = image.cuda(), label.cuda()
            image, label = Variable(image), Variable(label)
            
            optimizer.zero_grad()
    
            loss, R = model.objective(image, label)
    
            train_loss += loss.item()
            
            loss.backward()
            # step
            optimizer.step()
            
            t.update(1)
            
        train_loss /= len(train_loader)
        print(f'Epoch: {epoch}, Loss: {train_loss}')
        return train_loss


def test(phase = 'test'):
    
    model.eval()
    test_loss = 0
    
    if phase == 'test':
        print('Begin Testing Phase')
        loader = test_loader
        name_list = []
        label_list = []
        output_list = []
    elif phase == 'val':
        print('Begin Validation Phase')
        loader = val_loader
    # elif phase == 'priors':
    #     print('Loaded Priors set')
    #     loader = priors_loader
        

    for batch_idx, (image, label, name) in enumerate(loader):

        if args.cuda:
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)

        loss, R = model.objective(image, label)
        test_loss += loss.item()
        if phase == 'test':
            name_list.append(name[0])
            label_list.append(label.item())
            output_list.append(R.item())
        
    test_loss /= len(test_loader)
    print(f'Validation set, Loss: {test_loss}')
    if phase == 'test':
        return test_loss, name_list, label_list, output_list
    else:
        return test_loss
    
if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        device = torch.device('cuda')
        print('\nGPU is ON!')
        
    os.mkdir('./'+str(args.name))
        
    dataset = ProcasLoader(csf = args.csf, view_form = args.view, replicate = args.replicate)
    n_train = int(dataset.women_count*0.6)
    n_test  = int(dataset.women_count*0.25)
    n_val   = dataset.women_count - n_train - n_test
    print(f'Training on {n_train} women, Testing on {n_test} women')
    print(f'Validation on {n_val} women')
    
    
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    generator = torch.Generator().manual_seed(0)
    train_set, val_set, test_set = data_utils.random_split(range(dataset.women_count), [n_train, n_val, n_test], generator=generator)
    women_id = list(set(dataset.woman_list))
    
    train_set = [women_id[idx] for idx in train_set.indices]
    val_set = [women_id[idx] for idx in val_set.indices]
    test_set = [women_id[idx] for idx in test_set.indices]
    
    # print(f'Train set {train_set}')
    # print(f'Val set {val_set}')
    # print(f'Test set {test_set}')
    
    train_ids = np.array([x in train_set for x in dataset.woman_list])
    val_ids = np.array([x in val_set for x in dataset.woman_list])
    test_ids = np.array([x in test_set for x in dataset.woman_list])

    train_set = data_utils.Subset(dataset, np.where(train_ids)[0])
    val_set   = data_utils.Subset(dataset, np.where(val_ids)[0])
    test_set  = data_utils.Subset(dataset, np.where(test_ids)[0])

    
    train_loader = data_utils.DataLoader(train_set,
                                             batch_size=args.bs,
                                             shuffle=True,
                                             **loader_kwargs)
    val_loader = data_utils.DataLoader(val_set,
                                             batch_size=1,
                                             shuffle=False,
                                             **loader_kwargs)
    test_loader = data_utils.DataLoader(test_set,
                                             batch_size=1,
                                             shuffle=False,
                                             **loader_kwargs)
    model = Pvas_Model(pretrain = args.pretrained, replicate = args.replicate)
    if args.cuda:
        model.cuda()
    model = model.double()
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    stopper = EarlyStopper(patience = 5, min_delta = 0)
    
    
    epoch_losses = np.empty(args.epochs)
    val_losses = np.empty(args.epochs)
    stopping_epoch = 0
    min_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        epoch_losses[epoch-1] = train_loss
        val_loss = test(phase = 'val')
        val_losses[epoch-1] = val_loss
        out_data = pd.DataFrame({'Train_loss': epoch_losses, 
                                 'Val_loss':val_losses})
        out_data.to_csv('./'+str(args.name)+'/losses.csv', index = False)
        if val_loss < min_loss:
            print(f'Saving epoch {epoch} for having the lowest loss.')
            min_loss = val_loss
            torch.save(model.state_dict(),  './'+str(args.name)+'/model_pvas.pth')
        if stopper(val_loss):
            print(f'Training stopped during epoch {epoch} during to satisfying the stopping criterion')
            stopping_epoch = epoch
            break
    if args.cuda:
        model.load_state_dict(torch.load(  './'+str(args.name)+'/model_pvas.pth', map_location='cuda'))
    else:
        model.load_state_dict(torch.load(  './'+str(args.name)+'/model_pvas.pth', map_location='cpu'))    
    test_loss, name_list, label_list, output_list = test(phase = 'test')
    test_outdata = pd.DataFrame({'Names':name_list, 'Labels':label_list, 'Model_out': output_list})
    test_outdata.to_csv('./'+str(args.name)+'/test_out.csv', index = False)
    if not args.csf:
        plt.plot(epoch_losses, label = 'Train Loss')
        plt.plot(val_losses, label = 'Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train loss plot over epochs')
        plt.ylim([0, max(np.concatenate((epoch_losses, val_losses)))])
        plt.legend()
        plt.xticks(ticks = range(stopping_epoch), labels = range(1,stopping_epoch+1))
        plt.show()
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    
    
    # if args.priors:
    #     print('Begin priors testing')
    #     dataset_p = ProcasLoader(csf = args.csf, view_form = args.view, replicate = args.replicate, priors = True)
    #     priors_loader = data_utils.DataLoader(dataset_p,
    #                                          batch_size=1,
    #                                          shuffle=False,
    #                                          **loader_kwargs)
    #     test(phase = 'priors')
        