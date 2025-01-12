import os
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader

import sys
from utils.utils import read_pkl

DATA_FOLDER = os.environ.get('DATA_FOLDER', 'DataPath')

class ClinicalTtrialsPredictionDatasetH(Dataset):
    def __init__(self, DATA_FOLDER, subset, phase, max_length={'icds':5, 'smiless':5, 'in_criteria':5, 'ex_criteria':3}, reduce={'icds':'sum', 'criteria':'first'}):
        if phase not in ['I', 'II', 'III']:
            raise NotImplementedError
        
        if subset not in ['train', 'valid', 'merged', 'test']:
            raise NotImplementedError
        
        self.istrain = False
        if subset in ['train', 'valid', 'merged']:
            self.istrain = True
            
        if subset == 'train':
            DATA_FOLDER = f'{DATA_FOLDER}trainphase/phase_{phase}/'
        elif subset == 'valid':
            DATA_FOLDER = f'{DATA_FOLDER}validphase/phase_{phase}/'
        elif subset == 'merged':
            DATA_FOLDER = f'{DATA_FOLDER}mergedphase/phase_{phase}/'
        else:
            DATA_FOLDER  = f'{DATA_FOLDER}test/phase_{phase}/'

        self.files = glob(f'{DATA_FOLDER}*.pkl')


        if reduce['icds'] not in ['avg', 'sum'] or reduce['criteria'] not in ['first', 'avg', 'sum']:
            raise NotImplementedError
        
        self.reduce = reduce
        self.max_length = max_length
        

    def __len__(self):
        return len(self.files)
    
    def padding(self, embs, max_length):
        masks = [0 for _ in range(min(len(embs), max_length))]
        diff_length = max_length - len(embs)
        for _ in range(diff_length):
            embs.append(torch.zeros_like(embs[0]))
            masks.append(1)
        assert len(embs) == max_length
        assert len(masks) == max_length
        return embs, masks

    def truncat(self, embs, max_length):
        return embs[:max_length]
    
    # smiless only
    def scollect(self, embs, max_length):
        # print(len(embs))
        if len(embs) > max_length:
            embs = self.truncat(embs, max_length)
        tokens = []
        for emb in embs:
            token = (emb - torch.mean(emb)) / (torch.std(emb) + 1e-6)
            tokens.append(token)
        tokens, masks = self.padding(tokens, max_length)
        # print(tokens)
        return tokens, masks

    # icds only
    def icollect(self, embs, reduce, max_length):
        if len(embs) > max_length:
            embs = self.truncat(embs, max_length)
        tokens = []
        for emb in embs:
            if reduce == 'sum':
                token = torch.sum(emb, axis=0)
            elif reduce == 'avg':
                token = torch.mean(emb, axis=0)
            tokens.append(token)
        tokens, masks = self.padding(tokens, max_length)
        return tokens, masks
        
    
    def ccollect(self, embs, reduce, max_length):
        tokens = []
        if reduce == 'first':
            if isinstance(embs, list):
                embs = embs[0] * embs[1].unsqueeze(-1)
            for emb in embs:
                tokens.append(emb[0])


        elif reduce == 'sum' or reduce == 'avg':
            if isinstance(embs, list):
                num_non_padding_tokens = embs[1].sum(dim=1)
                # print('all_num', num_non_padding_tokens)
                embs = embs[0] * embs[1].unsqueeze(-1)
                for emb, num in zip(embs, num_non_padding_tokens):
                    # print('num', num)
                    if reduce == 'sum':
                        token = torch.sum(emb[1:num], axis=0)
                    elif reduce == 'avg':
                        token = torch.mean(emb[1:num], axis=0)
                    # print(token.size())
                    tokens.append(token)
            else:
                # print('Not a list')
                for emb in embs:
                    # print(emb.size())
                    if reduce == 'sum':
                        token = torch.sum(emb, axis=0)
                    elif reduce == 'avg':
                        token = torch.mean(emb, axis=0)
                    # print(token.size())
                    tokens.append(token)
        if len(tokens) > max_length:
            tokens = self.truncat(tokens, max_length)
        tokens, masks = self.padding(tokens, max_length)
        return tokens, masks



    def __getitem__(self, idx):
        pkl = read_pkl(self.files[idx])
        # print('nctid', pkl['nctid'])
        iembs = pkl['icdcodes']
        # print('iembs', iembs)
        itokens, imasks  = self.icollect(iembs, self.reduce['icds'], self.max_length['icds'])
        # print(itokens)

        # print(imasks)
        # print('****', len(iembs))
        sembs = pkl['smiless']
        # print('sembs', sembs)
        stokens, smasks = self.scollect(sembs, self.max_length['smiless'])
        # print(smasks)
        # print('****', len(sembs))
        cembs = pkl['criteria']
        # print('hellpo', len(cembs[0][1]))
        
        in_ctokens, in_cmasks = self.ccollect(cembs[0], self.reduce['criteria'], self.max_length['in_criteria'])
        # # print(in_cmasks)
        # # print('****', len(cembs[0]))
        ex_ctokens, ex_cmasks = self.ccollect(cembs[1], self.reduce['criteria'], self.max_length['ex_criteria'])
        # print(ex_cmasks)
        # print('****', len(cembs[1]))
        # print('***', len(itokens))
        # print('**', itokens[0].size())


        return {
            'itokens': torch.stack(itokens).to(torch.float32),
            'imasks': torch.tensor(imasks, dtype=torch.bool),
            'in_ctokens': torch.stack(in_ctokens).to(torch.float32),
            'in_cmasks': torch.tensor(in_cmasks, dtype=torch.bool),
            'ex_ctokens': torch.stack(ex_ctokens).to(torch.float32),
            'ex_cmasks': torch.tensor(ex_cmasks, dtype=torch.bool),
            'stokens': torch.stack(stokens).to(torch.float32),
            'smasks': torch.tensor(smasks, dtype=torch.bool),
            'label': torch.tensor(pkl['label'], dtype=torch.long),
            'nctid': pkl['nctid'],
        }



if __name__ == '__main__':
    dataset = ClinicalTtrialsPredictionDatasetH(DATA_FOLDER, 'test', 'III')
    dataloader = DataLoader(dataset , batch_size=1, shuffle=False, num_workers=0)
    imasks = torch.tensor([0,0,0,0,0])
    smasks = torch.tensor([0,0,0,0,0])
    in_cmasks = torch.tensor([0,0,0,0,0])
    ex_cmasks= torch.tensor([0,0,0])
    for d in dataloader:
        # print(d['itokens'].size())
        # print(d['imasks'])
        imasks += ~d['imasks'].squeeze()
        # print(imasks)
        # print(d['stokens'].size())
        # print(d['smasks'])
        smasks += ~d['smasks'].squeeze()
        # print(d['nctid'])
        # print(d['in_ctokens'].size())
        in_cmasks += ~d['in_cmasks'].squeeze()
        # print(d['in_cmasks'])
        # print(d['ex_ctokens'].size())
        # print(d['ex_cmasks'])
        ex_cmasks += ~d['ex_cmasks'].squeeze()

    print('imasks', imasks)
    print('smasks', smasks)
    print('in_cmasks', in_cmasks)
    print('ex_cmasks', ex_cmasks)
