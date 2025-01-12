import os
import pandas as pd
import torch
from tqdm import tqdm

DATA_PATH = os.environ.get('DATA_PATH', './data/clinicalTrialsPrediction/clinical-trial-outcome-prediction/data/raw_data.csv')

from utils.utils_whole import read_pkl, save_pkl


def get_icdcode(icdcode):
    itxt = icdcode[2:-2]
    code_lst = []
    for i in itxt.split('", "'):
        i = i[1:-1]
        # print([j.strip()[1:-1] for j in i.split(',')])
        code_lst.append([j.strip()[1:-1] for j in i.split(',')])
    return code_lst

def collect_icdcodes(icdcodes):
    code_set = set()
    for icdcode in icdcodes:
        code_lst = get_icdcode(icdcode)
        for codes in code_lst:
            merge_str = merge_code(codes, delimiter='~')
            code_set.add(merge_str)
    return code_set

def split_code(code):
    return code.split('~')

def merge_code(codes, delimiter='~'):
    merge_str = ''
    for code in codes:
        if len(merge_str):
            merge_str  = merge_str + delimiter
        merge_str = merge_str + code
    return merge_str

def code2embedding(code_set, embedding_size=64):
    from icdcodex import icd2vec, hierarchy
    embedder = icd2vec.Icd2Vec(num_embedding_dimensions=embedding_size, workers=-1)
    embedder.fit(*hierarchy.icd10cm())
    icdcodex_dict = {}
    for code in tqdm(code_set):
        # print(code)
        # print(split_code(code))
        # print(embedder.to_vec(split_code(code)))
        # print(torch.from_numpy(embedder.to_vec(split_code(code))))
        # print('~~~~')
        icdcodex_dict[code] = torch.from_numpy(embedder.to_vec(split_code(code)))
        save_pkl(icdcodex_dict, './iembed_dict.pkl')
    return icdcodex_dict
        
    

    


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    icdcodes = data['icdcodes']
    code_set = collect_icdcodes(icdcodes)
    print(len(code_set))
    save_pkl(code2embedding(code_set, embedding_size=64), './iembed_dict.pkl')
    # print(read_pkl('./iembed_dict.pkl'))

