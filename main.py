import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import sys
from ctp.evaluation import CTPEval
from ctp.dataset import ClinicalTtrialsPredictionDatasetH as CTPDataset
# from ctp.models import ClinicalTtrialsPredictionModelH as CTPModel
from utils.utils import random_seed, get_result_dir, create_result_dir, save_args, save_checkpoint

DATA_FOLDER = os.environ.get('DATA_FOLDER', 'DataPath')

parser = argparse.ArgumentParser(
        prog='ClinicalTtrialsPrediction Training',
        description='Training&Evaluation of CTP model.',
    )

parser.add_argument('--job', default='train', help='job', choices=['train', 'eval'])
parser.add_argument('--num_bootstraps', default=10, type=int, help='num_bootstraps')
parser.add_argument('--ratio_bootstraps', default=0.8, type=float, help='num_bootstraps')

# setting
parser.add_argument('--device', default='cuda', help='device', choices=['cuda', 'cpu'])
parser.add_argument('--seed', default=2023, help='seed', type=int)
parser.add_argument('--result_dir', default='./ab-results/', help='results dir')

# dataset
parser.add_argument('--subset', default='train', type=str, help='subset', choices=['train', 'vaild', 'merged'])
parser.add_argument('--phase', default='III', type=str, help='phase', choices=['I', 'II', 'III'])
parser.add_argument('--ireduce', default='sum', type=str, help='reduce', choices=['avg', 'sum'])
parser.add_argument('--creduce', default='first', type=str, help='reduce', choices=['first', 'avg', 'sum'])
parser.add_argument('--imax_length', default=5, type=int, help='max length of icds')
parser.add_argument('--smax_length', default=5, type=int, help='max length of smiless')
parser.add_argument('--in_cmax_length', default=5, type=int, help='max length of inclusion')
parser.add_argument('--ex_cmax_length', default=3, type=int, help='max length of exclusion')
parser.add_argument('--trainsize', default=64, type=int, help='train batch size')
parser.add_argument('--testsize', default=32, type=int, help='test batch size')


# models
parser.add_argument('--itoken_size', default=64, help='icdcodes', type=int)
parser.add_argument('--stoken_size', default=15, help='smiless', type=int)
parser.add_argument('--ctoken_size', default=768, help='criteria', type=int)
parser.add_argument('--dropout', default=0.005, help='dropout', type=float)
parser.add_argument('--nhead', default=2, help='nhead', type=int)
parser.add_argument('--nlayer', default=2, help='nlayer', type=int)
parser.add_argument('--emb_size', default=8, help='emb_size', type=int)
parser.add_argument('--epsilon', default=0.25, help='epsilon', type=float)
parser.add_argument('--temperature', default=0.2, help='temperature', type=float)
parser.add_argument('--rho1', default=5e-2, help='rho1', type=float)
parser.add_argument('--rho2', default=1e-2, help='rho2', type=float)
parser.add_argument('--threshold', default=0.3, help='threshold', type=float)

# training
parser.add_argument('--lr', default=5e-2, help='global learning rate', type=float)
parser.add_argument('--epochs', default=100, help='epoch', type=int)
parser.add_argument("--weighted", help="weighted loss", action="store_true")
parser.add_argument('--opt', default='adam', type=str, help='optimizer', choices=['adam', 'sgd'])


# evaluation
parser.add_argument('--metric', default='roc', help='metric for ranking', choices=['acc', 'f1', 'roc', 'pr'])
parser.add_argument('--model_path', help='model for evaluation', type=str)



def train_step(model, train_loader, optimizer, device):
    model.train()
    loss = 0
    bce = 0
    ntx = 0
    cauchy = 0
    correct = 0
    total = 0
    for data in tqdm(train_loader):
        itokens = data['itokens'].to(device)
        imasks = data['imasks'].to(device)
        stokens = data['stokens'].to(device)
        smasks = data['smasks'].to(device)
        in_ctokens = data['in_ctokens'].to(device)
        in_cmasks = data['in_cmasks'].to(device)
        ex_ctokens = data['ex_ctokens'].to(device)
        ex_cmasks = data['ex_cmasks'].to(device)
        labels = data['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(itokens, imasks, in_ctokens, in_cmasks, ex_ctokens, ex_cmasks, stokens, smasks, labels)

        outputs['loss'].backward()
        optimizer.step()

        with torch.no_grad():
            loss += outputs['loss'].data.item() * labels.size(0)
            bce += outputs['bceloss'].data.item() * labels.size(0)
            ntx += outputs['ntxloss'].data.item() * labels.size(0)
            cauchy += outputs['cauchyloss'].data.item() * labels.size(0)
            total += labels.size(0)
            decisions = (outputs['preds'] > 0).float()
            correct += (decisions == labels.float().view(-1, 1)).sum().item()
        # if total > 5:
        #     break
    
    return {
        'loss': loss / total,
        'bce': bce / total,
        'cauchy': cauchy / total,
        'ntx': ntx / total,
        'acc': 100. * correct / total
    }


def test_step(model, test_loader, device):
    model.eval()
    loss = 0
    bce = 0
    ntx = 0
    cauchy = 0
    correct = 0
    total = 0
    collect_predictions = []
    collect_decisions = []
    collect_labels = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            itokens = data['itokens'].to(device)
            imasks = data['imasks'].to(device)
            stokens = data['stokens'].to(device)
            smasks = data['smasks'].to(device)
            in_ctokens = data['in_ctokens'].to(device)
            in_cmasks = data['in_cmasks'].to(device)
            ex_ctokens = data['ex_ctokens'].to(device)
            ex_cmasks = data['ex_cmasks'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(itokens, imasks, in_ctokens, in_cmasks, ex_ctokens, ex_cmasks, stokens, smasks, labels)

            loss += outputs['loss'].data.item() * labels.size(0)
            bce += outputs['bceloss'].data.item() * labels.size(0)
            ntx += outputs['ntxloss'].data.item() * labels.size(0)
            cauchy += outputs['cauchyloss'].data.item() * labels.size(0)
            total += labels.size(0)
            decisions = (outputs['preds'] > 0).float()
            correct += (decisions == labels.float().view(-1, 1)).sum().item()
            
            collect_predictions.append(outputs['preds'].cpu().numpy())
            collect_decisions.append(decisions.cpu().numpy())
            collect_labels.append(labels.cpu().numpy())

            # if total > 5:
            #     break
    all_predictions = np.concatenate(collect_predictions)
    all_decisions =  np.concatenate(collect_decisions)
    all_labels = np.concatenate(collect_labels)

    f1, roc_auc, pr_auc = CTPEval(all_predictions, all_decisions, all_labels)
    return {
        'loss': loss / total,
        'bce': bce / total,
        'cauchy': cauchy / total,
        'ntx': ntx / total,
        'acc': 100. * correct / total,
        'f1': f1, 
        'roc': roc_auc, 
        'pr': pr_auc
    }


def eval_step(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    collect_predictions = []
    collect_decisions = []
    collect_labels = []
    with torch.no_grad():
        for data in test_loader:
            itokens = data['itokens'].to(device)
            imasks = data['imasks'].to(device)
            stokens = data['stokens'].to(device)
            smasks = data['smasks'].to(device)
            in_ctokens = data['in_ctokens'].to(device)
            in_cmasks = data['in_cmasks'].to(device)
            ex_ctokens = data['ex_ctokens'].to(device)
            ex_cmasks = data['ex_cmasks'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(itokens, imasks, in_ctokens, in_cmasks, ex_ctokens, ex_cmasks, stokens, smasks, labels)

            total += labels.size(0)
            decisions = (outputs['preds'] > 0).float()
            correct += (decisions == labels.float().view(-1, 1)).sum().item()

            collect_predictions.append(outputs['preds'].cpu().numpy())
            collect_decisions.append(decisions.cpu().numpy())
            collect_labels.append(labels.cpu().numpy())

            # if total > 5:
            #     break
    all_predictions = np.concatenate(collect_predictions)
    all_decisions =  np.concatenate(collect_decisions)
    all_labels = np.concatenate(collect_labels)

    f1, roc_auc, pr_auc = CTPEval(all_predictions, all_decisions, all_labels)
    return {
        'f1': f1, 
        'roc': roc_auc, 
        'pr': pr_auc
    }


def main():
    args = parser.parse_args()
    random_seed(args.seed)
    if args.job == 'train':
        result_dir = get_result_dir(args)
        result_dir = create_result_dir(result_dir)

        save_args(args, result_dir)

        max_length = {
            'icds':args.imax_length,
            'smiless':args.smax_length,
            'in_criteria':args.in_cmax_length,
            'ex_criteria':args.ex_cmax_length
        }

        reduce = {
            'icds': args.ireduce,
            'criteria': args.creduce,
        }

        train_dataset = CTPDataset(DATA_FOLDER, subset=args.subset, phase=args.phase, max_length=max_length, reduce=reduce)
        test_dataset = CTPDataset(DATA_FOLDER, subset='test', phase=args.phase, max_length=max_length, reduce=reduce)
        
        train_loader = DataLoader(train_dataset , batch_size=args.trainsize, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_dataset , batch_size=args.testsize, shuffle=False, num_workers=1)

        if args.nlayer == 0:
            NotImplementedError
        else:
            from ctp.models_nlayers import ClinicalTtrialsPredictionModelH_nlayers as CTPModel
            print('nlayers', args.nlayer)

        model = CTPModel(args, max_length=max_length)
        model.to(args.device)



        if args.opt == 'adam':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.01)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch / args.warmup_epochs) if epoch < args.warmup_epochs else max(0.0, (args.epochs - epoch) / (args.epochs - args.warmup_epochs)))
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)

        print('ok')

        dfhistory = pd.DataFrame(
            columns=[
                'epoch', 'train_loss', 'train_bce', 'train_cauchy', 'train_ntx', 'train_acc',
                'test_loss', 'test_bce', 'test_cauchy', 'test_ntx', 'test_acc', 
                'test_f1', 'test_roc', 'test_pr',
            ], 
            dtype=np.float16
        )

        best = 0
        
        for epoch in range(1, 1 + args.epochs):
            print('Training...')
            train = train_step(model, train_loader, optimizer, args.device)
            print('Test...')
            test = test_step(model, test_loader, args.device)

            info = (
                int(epoch), train['loss'], train['bce'], train['cauchy'], train['ntx'], train['acc'],
                test['loss'],test['bce'], test['cauchy'], test['ntx'], test['acc'], 
                test['f1'], test['roc'], test['pr'],
            )
            dfhistory.loc[epoch-1] = info
            dfhistory.to_csv(f'{result_dir}/history.csv', index=False)
            
            isbest = test[args.metric] > best
            best = max(best, test[args.metric])
            
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'stastics': {'train':train, 'test': test}
            }
            
            save_checkpoint(state, isbest, args.metric, epoch, result_dir)

            print(f'Epoch={epoch} done!')
            # scheduler.step()


    elif args.job == 'eval':

        epoch = args.model_path.split('@')[-1].split('.')[0]
        json_path = args.model_path.split('checkpoints')[0] + f'mun{args.num_bootstraps}_ratio{args.ratio_bootstraps}_model@{epoch}.json'
        phase = args.model_path.split('phase_')[1].split('/')[0]

        from utils.utils import read_json, save_json
        # if len(glob(json_path)) > 0:
        #     read_json(json_path)
        #     return
        
        max_length = {
            'icds':args.imax_length,
            'smiless':args.smax_length,
            'in_criteria':args.in_cmax_length,
            'ex_criteria':args.ex_cmax_length
        }

        reduce = {
            'icds': args.ireduce,
            'criteria': args.creduce,
        }


        test_dataset = CTPDataset(DATA_FOLDER, subset='test', phase=phase, max_length=max_length, reduce=reduce)
        

        if args.nlayer == 0:
            NotImplementedError
        else:
            from ctp.models_nlayers import ClinicalTtrialsPredictionModelH_nlayers as CTPModel
            print('nlayers', args.nlayer)

        model = CTPModel(args)
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
        model.to(args.device)
        statistics = {}
        statistics['f1'] = []
        statistics['roc'] = []
        statistics['pr'] = []
        print(args.num_bootstraps)
        for _ in range(args.num_bootstraps):
            indices = np.random.choice(len(test_dataset), int(args.ratio_bootstraps*len(test_dataset)), replace=True)
            # print(indices)
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.testsize,
                    shuffle=False,
                    sampler=sampler
                )
            eval = eval_step(model, test_loader, args.device)
            statistics['f1'].append(eval['f1'])
            statistics['roc'].append(eval['roc'])
            statistics['pr'].append(eval['pr'])
        f1 = np.percentile(statistics['f1'], [2.5, 97.5])
        roc = np.percentile(statistics['roc'], [2.5, 97.5])
        pr = np.percentile(statistics['pr'], [2.5, 97.5])
        statistics['f1_2.5'] = f1[0]
        statistics['f1_97.5'] = f1[1]
        statistics['roc_2.5'] = roc[0]
        statistics['roc_97.5'] = roc[1]
        statistics['pr_2.5'] = pr[0]
        statistics['mean_f1'] = np.mean(statistics['f1'])
        statistics['std_f1'] = np.std(statistics['f1'])
        statistics['mean_pr'] = np.mean(statistics['pr'])
        statistics['std_pr'] = np.std(statistics['pr'])
        statistics['mean_roc'] = np.mean(statistics['roc'])
        statistics['std_roc'] = np.std(statistics['roc'])
        save_json(statistics, json_path)









if __name__ == '__main__':
    main()
