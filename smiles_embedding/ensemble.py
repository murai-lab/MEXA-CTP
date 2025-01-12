import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

DATA_PATH = os.environ.get('DATA_PATH', '/home/yzhang37/data/clinicalTrialsPrediction/clinical-trial-outcome-prediction/data/raw_data.csv')

import sys
sys.path.insert(0, '/home/yzhang37/Trials/')
from utils.utils_whole import read_pkl, save_pkl


def get_smiles(smiles):
    stxt = smiles[1:-1]
    # print([i.strip()[1:-1] for i in stxt.split(',')])
    return [i.strip()[1:-1] for i in stxt.split(',')]

def collect_smiless(smiless):
    smiless_set = set()
    for smiles in smiless:
        smiles_lst = get_smiles(smiles)
        for smile in smiles_lst:
            # print(smile)
            smiless_set.add(smile)
    return smiless_set

# def smiless2embedding(smiless_set, embedding_size=15):
#     smiless_lst = list(smiless_set)
#     from macaw import MACAW
#     mcw = MACAW(n_components=embedding_size)
#     mcw.fit(smiless_set)
#     embs = mcw.transform(smiless_set)
#     mcw_dict = {}
#     count = 0
#     for emb, smiles in zip(embs, smiless_lst):
#         if np.sum(np.isnan(emb)) > 1:
#             print(emb)
#             count += 1
#             print(emb.shape)
#             mcw_dict[smiles] = torch.zeros_like(torch.from_numpy(emb))
#         else:
#             mcw_dict[smiles] = torch.from_numpy(emb)
#     print(count)
#     return mcw_dict

def smiless2embedding(smiless_set):
    smiless_lst = list(smiless_set)
    import deepchem as dc
    featurizer = dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False)
    cmf_dict = {}
    count = 0
    for smiles in smiless_lst:
        emb = featurizer.featurize(smiles)
        print(dir(emb))
        print(emb.data())
        try:
            emb = featurizer.featurize(smiles)
            print(emb)
            cmf_dict[smiles] =  torch.from_numpy(emb)
        except:
            cmf_dict[smiles] =  torch.zeros_like(torch.from_numpy(emb))

    print(count)
    return cmf_dict



if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    smiless = data['smiless']
    smiless_set = collect_smiless(smiless)
    print(len(smiless_set))
    a = smiless2embedding(smiless_set)
    save_pkl(a, './sembed_dict.pkl')