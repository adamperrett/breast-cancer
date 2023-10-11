import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler


from pvas_loader import ProcasLoader
from model import Pvas_Model
from pvas_utils import EarlyStopper


def get_args():
    parser = argparse.ArgumentParser(description='Training pVAS sr')
    parser.add_argument('--csf', action='store_true', default=False, help='Are we training on the csf or not?')
    parser.add_argument('--no-cuda', action='store_true', default = False, help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR')
    parser.add_argument('--bs', type=int, default=4, metavar='B')
    parser.add_argument('--view', type=str, default='CC', help = 'Which mammographic view to use')
    parser.add_argument('--pretrained', action='store_true', default=False, help = 'Use pretrained network or not?')
    parser.add_argument('--replicate', action='store_true', default=True, help = 'Replicate or not? IF no will replace the first CNN layer.')
    parser.add_argument('--priors', action='store_true', default=False, help = 'Test on the priors automatically on the best model?')
    return parser.parse_args()

def train(epoch, model, optimizer, train_loader):
    model.train()
    train_loss = 0
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
        
    train_loss /= len(train_loader)
    print(f'Epoch: {epoch}, Loss: {train_loss}')
    return train_loss


def test(model, phase = 'test'):
    
    model.eval()
    test_loss = 0
    if phase == 'test':
        print('Loaded Test set')
        loader = test_loader
    elif phase == 'val':
        print('Loaded Val set')
        loader = val_loader
    for batch_idx, (image, label, name) in enumerate(loader):

        if args.cuda:
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)

        loss, R = model.objective(image, label)
        test_loss += loss.item()

        
    test_loss /= len(test_loader)
    print(f'Validation set, Loss: {test_loss}')
    return test_loss

def train_function(config):
    model = Pvas_Model(pretrain = config['pretrain'], replicate = args.replicate)
    if args.cuda:
        model.cuda()
    model = model.double()
       
    train_loader = data_utils.DataLoader(train_set,
                                             batch_size=config['batch_size'],
                                             shuffle=True,
                                             **loader_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
    
    
    epoch_losses = np.empty(args.epochs)
    val_losses = np.empty(args.epochs)
    stopping_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, optimizer, train_loader)
        epoch_losses[epoch-1] = train_loss
        val_loss = test(model, phase = 'val')
        val_losses[epoch-1] = val_loss
        session.report({"val_loss": val_loss})
    session.report({"val_loss": val_losses.min() })
    
    
if __name__ == "__main__":
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        device = torch.device('cuda')
        print('\nGPU is ON!')
    dataset = ProcasLoader(csf = args.csf, view_form = args.view, replicate = args.replicate)
    n_train = int(dataset.women_count*0.6)
    n_test  = int(dataset.women_count*0.25)
    n_val   = dataset.women_count - n_train - n_test
    print(f'Training on {n_train} women, Testing on {n_test} women')
    print(f'Validation on {n_val} women')
    
    
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    generator = torch.Generator().manual_seed(10) #this seed for testing
    train_set, val_set, test_set = data_utils.random_split(
        range(dataset.women_count), [n_train, n_val, n_test], generator=generator)
    women_id = list(set(dataset.woman_list))
    
    train_set = [women_id[idx] for idx in train_set.indices]
    val_set = [women_id[idx] for idx in val_set.indices]
    test_set = [women_id[idx] for idx in test_set.indices]

    
    train_ids = np.array([x in train_set for x in dataset.woman_list])
    val_ids = np.array([x in val_set for x in dataset.woman_list])
    test_ids = np.array([x in test_set for x in dataset.woman_list])

    train_set = data_utils.Subset(dataset, np.where(train_ids)[0])
    val_set   = data_utils.Subset(dataset, np.where(val_ids)[0])
    test_set  = data_utils.Subset(dataset, np.where(test_ids)[0])

    
    
    val_loader = data_utils.DataLoader(val_set,
                                             batch_size=1,
                                             shuffle=False,
                                             **loader_kwargs)
    test_loader = data_utils.DataLoader(test_set,
                                             batch_size=1,
                                             shuffle=False,
                                             **loader_kwargs)
    ray.shutdown()
    ray.init()
    
    config = {
        'learning_rate':tune.loguniform(0.00001, 0.01),
        "batch_size": tune.choice([2, 6, 10, 14]),
        'pretrain': tune.choice([True])
        }
    scheduler = ASHAScheduler(metric = 'val_loss',
                              mode = 'min',
                              grace_period = 4) 
    resources_per_trial = {"cpu": 1, "gpu": 1}

    
    tuner = tune.run(
        train_function,
        resources_per_trial=resources_per_trial,
        config=config,
        num_samples=10,
        scheduler=scheduler,
        time_budget_s = 60 * 60 * 24 * 3 + 60 * 60 * 8
        )
    
    results = tuner.results
    print('All configurations are: ',tuner.get_all_configs())
    print('Best configuration is achieved by: ', tuner.get_best_config(metric  = 'val_loss', mode = 'min'))
    ray.shutdown()