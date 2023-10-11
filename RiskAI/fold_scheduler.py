import subprocess
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from pretrain_patcher2 import ProcasBags

def submit_job(script_name):
    job_command = f'qsub {script_name}'
    subprocess.run(job_command, shell = True)
    
def create_script(fold, args):
    with open(f'./{args.name}/fold_{fold+1}/{args.name}_script_{fold+1}', 'w') as file:
        file.write('#!/bin/bash --login\n')
        file.write('\n')
        file.write('#$ -cwd\n')
        file.write('#$ -l nvidia_v100=1\n')
        file.write('\n')
        file.write('conda activate ml\n')
        file.write('\n')
        ## Change this back to run_fold
        file.write(f'python run_fold_notest.py --epochs 25 --csf {args.csf} --lr 0.000007 --repeats 5 --name {args.name} --pretrain y --priors y --type proc --backbone res --attention y --partition {args.partition} --fold {fold}')


def get_args():
    parser = argparse.ArgumentParser(description='Master script')
    parser.add_argument('--folds', type=int, default=5, metavar='N',help='Number of folds, defaults to 5')
    parser.add_argument('--name', type=str, help='The name of the expriment')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)') 
    parser.add_argument('--csf', type=str, default='y', help='Is this ran on the csf or not y/n?')
    parser.add_argument('--partition', type=str, default ='sdcs', help = 'Which slice of the data to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    Path(f'./{args.name}').mkdir(parents=True, exist_ok=True)
    
    
    side_seed = args.seed
    dataset_t = ProcasBags(train=True,csf = args.csf, partition = args.partition, tile_size = 224, seed = side_seed)
    patients = np.array(np.unique([row[1] for row in dataset_t.mamm_list]))
    
    splits=KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx,test_idx) in enumerate(splits.split(np.arange(len(patients)))):
        Path(f'./{args.name}/fold_{fold+1}').mkdir(parents=True, exist_ok=True)
        create_script(fold, args)
        
        
        train_patients = patients[train_idx]
        test_patients = patients[test_idx]
        
        
        with open(Path(f'./{args.name}/fold_{fold+1}/train_set_{fold+1}.txt'), 'w') as train_file:
            train_file.write('\n'.join(train_patients))
            
        with open(Path(f'./{args.name}/fold_{fold+1}/test_set_{fold+1}.txt'), 'w') as test_file:
            test_file.write('\n'.join(test_patients))
        submit_job(f'{args.name}/fold_{fold+1}/{args.name}_script_{fold+1}')
        
