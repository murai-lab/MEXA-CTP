import os
import torch

import sys
sys.path.insert(0, '/home/yzhang37/Trials/')
from utils.embedding import Embedding as EMB
from utils.utils_whole import save_pkl, read_pkl
from ensemble import *

DATA_PATH = os.environ.get('DATA_PATH', '/home/yzhang37/data/clinicalTrialsPrediction/clinical-trial-outcome-prediction/data/raw_data.csv')

class icdsEmbedding(EMB):
    def __init__(self, DATA_PATH, attribute, output_path, embedding_size=2):
        super(icdsEmbedding, self).__init__(DATA_PATH, attribute, output_path)
        self.embedding_size = embedding_size
        self.icdcodex_dict = read_pkl('./iembed_dict.pkl')


    def clean_step(self, idx):
        itxt = self.content[idx][2:-2]
        code_lst = []
        for i in itxt.split('", "'):
            i = i[1:-1]
            code_lst.append([j.strip()[1:-1] for j in i.split(',')])
        print(code_lst)
        return code_lst

    def embed_step(self, clean_data):
        return self.icdcodex(clean_data)
    
    def icdcodex(self, data):
        iembs = []
        for d in data:
            key = merge_code(d, delimiter='~')
            iembs.append(self.icdcodex_dict[key])
        print(iembs)
        return iembs

    def save(self, clean_data, idx, path):
        clean_data_dict = {}
        clean_data_dict['label'] = self.label[idx]
        clean_data_dict['nctid'] = self.nctids[idx]
        clean_data_dict['icdcodes'] = clean_data
        save_pkl(clean_data_dict, path + self.nctids[idx] + '.pkl')
        return
    
if __name__ == '__main__':
    ie = icdsEmbedding(DATA_PATH, 'icdcodes', '../data/')
    ie.worker()
    # ie.clean_step(1)