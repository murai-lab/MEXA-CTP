import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import sys
sys.path.insert(0, '/home/yzhang37/Trials/')
from utils.utils_whole import create_path, read_pkl, save_pkl

FILENAME_FOLDER = os.environ.get('FILENAME_FOLDER', '/home/yzhang37/data/clinicalTrialsPrediction/clinical-trial-outcome-prediction/data/')
DATA_FOLDER = os.environ.get('DATA_FOLDER', '/home/yzhang37/Trials/data/')

def create_subset(nctids, emb_path, data_paths):
    create_path(emb_path)
    for nctid in tqdm(nctids):
        save_pkl(merge_file(nctid, data_paths), f'{emb_path}{nctid}.pkl')
        # print(read_pkl(f'{emb_path}{nctid}.pkl'))
        # move_file(f'{DATA_FOLDER}{attribute}/embed/{nctid}.pkl', f'{emb_path}{nctid}.pkl')

def merge_file(nctid, data_folders):
    merge_dict = {}
    merge_dict['nctid'] = nctid
    for data_folder in data_folders:
        attribute = data_folder.split('/')[-2]
        pkl = read_pkl(f'{data_folder}embed/{nctid}.pkl')
        if attribute == 'icdcodes':
            merge_dict['label'] = pkl['label']
        merge_dict[attribute] = pkl[attribute]
    # print('-----')
    # print(merge_dict)
    return merge_dict

def create_dataset(DATA_FOLDER, FILENAME_FOLDER, output_path):
    DATA_FOLDERs = []  
    for attribute in ['icdcodes', 'smiless', 'criteria']:
        DATA_FOLDERs.append(f'{DATA_FOLDER}{attribute}/') 
    
    # istest
    for phase in ['I', 'II', 'III']:
        test_nctids = np.array(pd.read_csv(f'{FILENAME_FOLDER}phase_{phase}_test.csv')['nctid'])
        print(len(test_nctids))
        emb_path = output_path + f'test/phase_{phase}/'
        create_subset(test_nctids, emb_path, DATA_FOLDERs)

    for phase in ['I', 'II', 'III']:
        train_nctids = np.array(pd.read_csv(f'{FILENAME_FOLDER}phase_{phase}_train.csv')['nctid'])
        print(len(train_nctids))
        emb_path = output_path + f'trainphase/phase_{phase}/'
        create_subset(train_nctids, emb_path, DATA_FOLDERs)

    for phase in ['I', 'II', 'III']:
        valid_nctids = np.array(pd.read_csv(f'{FILENAME_FOLDER}phase_{phase}_valid.csv')['nctid'])
        print(len(valid_nctids))
        emb_path = output_path + f'validphase/phase_{phase}/'
        create_subset(valid_nctids, emb_path, DATA_FOLDERs)
    

    return

if __name__ == '__main__':
    create_dataset(DATA_FOLDER, FILENAME_FOLDER, output_path='./data/')
