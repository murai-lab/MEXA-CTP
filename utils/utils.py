import os
import json
import torch
import pickle
import warnings
import numpy as np
import torch.nn as nn

def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    warnings.warn('You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

def save_pkl(data_dict, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pkl(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

def save_json(data_dict, path):
    with open(path, 'w') as json_file:
        json.dump(data_dict, json_file)

def read_json(path):
    with open(path, 'r') as json_file:
        loaded_dict = json.load(json_file)
    print(loaded_dict)
    
def path_exist(path):
    return os.path.exists(path)

def create_path(path):
    if not path_exist(path):
        os.makedirs(path)
        return

def get_result_dir(args):
    result_dir = args.result_dir
    result_dir = result_dir + f'{args.subset}_phase_{args.phase}/'
    if args.weighted:
        result_dir = result_dir + 'wei_'
    if args.opt == 'adam':
        result_dir = result_dir + f'rho1{args.rho1}_rho2{args.rho2}'
    elif args.opt == 'sgd':
        result_dir = result_dir + f'sgdw_lr{args.lr}'
    return result_dir

def create_result_dir(result_dir):
    id = 0
    while True:
        result_dir_id = result_dir + '_id%d'%id
        if not path_exist(result_dir_id): break
        id += 1
    os.makedirs(result_dir_id)
    os.makedirs(result_dir_id + '/checkpoints')
    return result_dir_id

def save_args(args, result_dir):
    args_dict = vars(args)
    filename = result_dir + '/args.json'
    with open(filename, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


def findcriterion(weighted, phase, device):
    if not weighted:
        return nn.BCEWithLogitsLoss() 
    else:
        if phase == 'I':
            print('weighted')
            return nn.BCEWithLogitsLoss(torch.Tensor([452/592]).to(device))
        elif phase == 'II':
            return nn.BCEWithLogitsLoss(torch.Tensor([2091/1914]).to(device))
        elif phase == 'III':
            return nn.BCEWithLogitsLoss(torch.Tensor([1073/2021]).to(device))
        else:
            raise NotImplementedError


def save_checkpoint(state, isbest, metric, epoch, result_dir):
    marked = metric
    print(isbest)
    if isbest:
        filename = result_dir + f'/checkpoints/{marked}best@{epoch}.pth.tar'
    else:
        filename = result_dir + f'/checkpoints/model@{epoch}.pth.tar'
    torch.save(state, filename)
    return
