from __future__ import print_function

import numpy as np
import pandas as pd
from pathlib import Path
import os

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve
import math
import random
import openpyxl
from sklearn.model_selection import KFold
from GPUtil import showUtilization as gpu_usage


from pretrain_patcher2 import ProcasBags

from pre_model import Pretr_Att

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test_per', type=float, default=0.8, help='Percent of examples for testing.')
    parser.add_argument('--csf', type=str, default='n', help='Is this ran on the csf or not y/n?')
    parser.add_argument('--repeats', type=int, default=1, help='Number of repeats')
    parser.add_argument('--save', type=str, default='n', help='Save the models or not')
    parser.add_argument('--name', type=str, help='The AUC name')
    parser.add_argument('--pretrain', type=str, default='n', help='The AUC name')
    parser.add_argument('--priors', type=str, default='n', help='Test on priors?')
    parser.add_argument('--type', type=str, default='hproc', help='Type of data to load')
    parser.add_argument('--backbone', type=str, help='Type of backbone')
    parser.add_argument('--attention', type=str, default='y', help = 'Type of aggregation used')  
    parser.add_argument('--fold', type=int, help = 'What fold is being run')
    parser.add_argument('--partition', type=str, default ='sdcs', help = 'Which slice of the data to use')
    return parser.parse_args()

args = get_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print('\nGPU is ON!')


print(f'Running job for fold number {args.fold}')


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    probs = []
    label_list = []
    pred_label = []
    names = []
    
    #data shape (num_tiles,3,224,224)
    for batch_idx, (data, label, name) in enumerate(train_loader):
        #print(data.shape)
        bag_label = label[0]
        names.append(name[0])
        label_list.append(bag_label.item()) 
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        torch.cuda.empty_cache()
        
        #print(torch.cuda.memory_summary(device=device))
        errors, predicted_label, prob, loss = model.calculate_objective(data, bag_label)
        
        #print(torch.cuda.memory_summary(device=device))
        train_loss += loss.data[0].item()

        train_error += errors
        probs.append(prob.item())
        pred_label.append(predicted_label.item())
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        
        #print(torch.cuda.memory_summary(device=device))
    print('Train')
    dic = {'Name': names, 'True': label_list, 'Pred':pred_label, 'Prob': probs}
    df = pd.DataFrame(dic)
    # df.to_csv(str(epoch)+'logits_under.csv', encoding='utf-8')
    #print('Bag:'+str(label_list))
    #print('Pred:' +str(pred_label))
    #print('Prob'+str(probs))    
        
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    auc = roc_auc_score(label_list, probs)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, AUC: {:.4f}'.format(epoch, train_loss, train_error, auc))
    return auc, train_loss, df
    # Saves the first, last and every 10th checkpoints
    
def test(epoch,repeat, max_auc, final_auc,  validation=False, contra = False, priors = False):
    model.eval()
    test_loss = 0.
    test_error = 0.
    probs = []
    label_list = []
    pred_label = []
    names = []
    with torch.no_grad():
        if validation:
            loader = val_loader
        elif priors:
            loader = prior_loader
        else:
            loader = test_loader
        for batch_idx, (data, label, name) in enumerate(loader):
            names.append(name[0])
            bag_label = label[0]
            label_list.append(bag_label.item())
            instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            
            errors, predicted_label, prob, loss = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0].item()
            pred_label.append(predicted_label.item())
            test_error += errors
            probs.append(prob.item())
    dic = {'Name': names, 'True': label_list, 'Pred':pred_label, 'Prob': probs}
    df = pd.DataFrame(dic)
    # df.to_csv(str(epoch)+'logits_val_under.csv', encoding='utf-8')
    #print('Bag:'+str(label_list))
    #print('Pred:' +str(pred_label))
    #print('Prob'+str(probs))
    auc = roc_auc_score(label_list, probs)
    print('The auc is :'+str(auc))
        
    

    test_error /= len(loader)
    test_loss /= len(loader)
    
    if validation:
        print('\nValidation Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
        #print('\nValidation Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy(), test_error))
    else:
        print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
    if args.priors == 'y':
        if max_auc < auc:
            torch.save(model.state_dict(),  f'./{args.name}/fold_{args.fold+1}/' + str(repeat) + '_fold.pth')
            print('Saved epoch {} based on performance'.format(epoch))
    return auc, test_loss, df

if __name__ == "__main__":
    full_dic = {}
    side_seed = args.seed
    dataset_t = ProcasBags(train=True,csf = args.csf, partition = args.partition, tile_size = 224, seed = side_seed)
    dataset_v = ProcasBags(train=False,csf = args.csf, partition = args.partition, tile_size = 224, seed = side_seed)
    dataset_p = ProcasBags(train=False,csf = args.csf, partition = 'priors', tile_size = 224, seed = side_seed)
    with open(f'./{args.name}/fold_{args.fold+1}/train_set_{args.fold+1}.txt', 'r') as train_file:
        train_samples = train_file.readlines()
        train_samples = [line.strip() for line in train_samples]
    with open(f'./{args.name}/fold_{args.fold+1}/test_set_{args.fold+1}.txt', 'r') as test_file:
        test_samples = test_file.readlines()
        test_samples = [line.strip() for line in test_samples]
    
    
    patients = np.array(np.unique([row[1] for row in dataset_t.mamm_list]))
    output_logits = pd.DataFrame()
    train_logits = pd.DataFrame()
    contra_logits = pd.DataFrame()
    prior_logits = pd.DataFrame()
    final_auc = 0
    repeats = args.repeats
    repeat_epoch = 4

    

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_ind = np.array([row[1] in train_samples for row in dataset_t.mamm_list])
    test_ind = np.array([row[1] in test_samples for row in dataset_t.mamm_list])
    #test_sampler = data_utils.SubsetRandomSampler(test_ind)
    
    
    patients_val = train_samples
    print('Patient check: ', patients_val[0:2])
    n_train = int(len(patients_val)*0.85)
    n_val   = len(patients_val) - n_train
    random.seed(args.seed)
    train_pat = random.sample(set(patients_val), n_train)
    #print(train_pat)
    val_pat = list(set(patients_val) - set(train_pat))
    #print(val_pat)
    
    
    train_ind = np.array([row[1] in train_pat for row in dataset_t.mamm_list])

    val_ind = np.array([row[1] in val_pat for row in dataset_t.mamm_list])
    #print([row[1] for row in dataset_t.mamm_list])
    #print(train_ind)
    #print(np.where(train_ind)[0])
    print('Train amount: ', train_ind.sum())
    print('Val amount: ', val_ind.sum())
    print('Test amount: ', test_ind.sum())
    train_set = data_utils.Subset(dataset_t, np.where(train_ind)[0])
    val_set   = data_utils.Subset(dataset_v, np.where(val_ind)[0])
    test_set  = data_utils.Subset(dataset_v, np.where(test_ind)[0])

    sample_weights = np.take([row[2] for row in dataset_t.mamm_list], np.where(train_ind)[0])
    #print(sample_weights)
    sample_weights = [1/75 if x==0 else 1/25 for x in sample_weights]
    #print(sample_weights)
    train_loader = data_utils.DataLoader(train_set,
                                          batch_size=1,
                                          sampler = data_utils.WeightedRandomSampler(weights = sample_weights, num_samples=int(len(train_set)), replacement=True),
                                          **loader_kwargs)
    # train_loader = data_utils.DataLoader(train_set,
    #                                       batch_size=1,
    #                                       shuffle = True,
    #                                       **loader_kwargs)
    
    
    
    
    val_loader = data_utils.DataLoader(val_set,
                                    batch_size=1,
                                    shuffle=False,
                                     **loader_kwargs)
    test_loader = data_utils.DataLoader(test_set,
                                    batch_size=1,
                                    shuffle=False,
                                      **loader_kwargs)

    prior_loader = data_utils.DataLoader(dataset_p,
                                    batch_size=1,
                                    shuffle=False,
                                      **loader_kwargs)
    train_e = []
    test_e = []
    train_l = []
    test_l = []
    max_auc = 0 
    print('Repeat finder')
    repeat_auc = []
    # dir_checkpoint = Path('./'+args.name+'_model/')
    # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        if args.attention == 'y':
            model = Pretr_Att(args.pretrain, args.backbone)
        else:
            model = Pretr_Att(args.pretrain, args.backbone, attention_flag = False)
        print('The learning rate is:' + str(args.lr))
        if args.pretrain == 'y':
            print('Model pretrained')
        elif args.pretrain == 'n':
            print('Model untrained')
        
        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        
        
        for epoch in range(1, repeat_epoch+1):
            train_auc, train_loss, logits = train(epoch)
            test_auc, test_loss, logits_val = test(epoch, int(args.fold)+1, 1, 1, validation=True)
        repeat_auc.append(test_auc)
        torch.save(model.state_dict(), Path(f'./{args.name}/fold_{args.fold+1}/', str(repeat) + 'repeat.pth'))
    print('Repeat AUCs: ', repeat_auc)
    
    if args.cuda:
        model.load_state_dict(torch.load(f'./{args.name}/fold_{args.fold+1}/' +  str(np.argmax(repeat_auc))+'repeat.pth', map_location='cuda'))
    else:
        model.load_state_dict(torch.load(f'./{args.name}/fold_{args.fold+1}/' +  str(np.argmax(repeat_auc))+'repeat.pth', map_location='cpu'))   
    print('Loaded repeat: ',str(np.argmax(repeat_auc)+1))    
    print('Begin training Phase')    
    for epoch in range(1, args.epochs + 1):
        train_auc, train_loss, logits = train(epoch)
        train_logits = pd.concat([train_logits, logits])
        train_logits.to_csv(Path(f'./{args.name}/fold_{args.fold+1}/'+args.name+str(args.fold+1)+'_train.csv'),  encoding='utf-8', index = False)
        print('Start Validation Round')
        test_auc, test_loss, logits_val = test(epoch, int(args.fold)+1, max_auc, final_auc, validation=True)
        max_auc = max(max_auc, test_auc)
        final_auc = max(final_auc, test_auc)
        train_e.append(train_auc)
        test_e.append(test_auc)
        train_l.append(train_loss)
        test_l.append(test_loss)
        auc_dic = {'train_auc':train_e, 'test_auc':test_e, 'train_loss':train_l, 'test_loss':test_l}
        df1 = pd.DataFrame(auc_dic)
        df1.to_csv(Path(f'./{args.name}/fold_{args.fold+1}/'+args.name+str(args.fold+1)+'_stats.csv'), encoding='utf-8')
        print('Saved in:', args.name)
    print('The max AUC is: ', max_auc)
    if args.priors == 'y':
        if args.cuda:
            model.load_state_dict(torch.load(  f'./{args.name}/fold_{args.fold+1}/' + str(int(args.fold)+1) + '_fold.pth', map_location='cuda'))
        else:
            model.load_state_dict(torch.load(  f'./{args.name}/fold_{args.fold+1}/' + str(int(args.fold)+1) + '_fold.pth', map_location='cpu'))
        print('Begin Domain Testing Phase')
        test_auc, test_loss, logits_test = test(0, int(args.fold)+1, 1,1, validation=False)
        logits_test.to_csv(Path(f'./{args.name}/fold_{args.fold+1}/'+args.name+str(args.fold+1)+'_domain.csv'), encoding='utf-8', index = False)
        print('Saved in: '+args.name+'_test.csv')
        
        print('Begin Target Testing Phase')
        test_auc, test_loss, logits_test = test(0, int(args.fold)+1, 1,1, priors = True)
        logits_test.to_csv(Path(f'./{args.name}/fold_{args.fold+1}/'+args.name+str(args.fold+1)+'_target.csv'), encoding='utf-8', index = False)
print('best final AUC is:', final_auc)
for repeat in range(repeats):
    os.remove(Path(f'./{args.name}/fold_{args.fold+1}/', str(repeat) + 'repeat.pth'))