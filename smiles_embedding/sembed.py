import os
import torch

from utils.embedding import Embedding as EMB
from utils.utils import save_pkl, read_pkl

DATA_PATH = os.environ.get('DATA_PATH', './data/clinicalTrialsPrediction/clinical-trial-outcome-prediction/data/raw_data.csv')

class smilessEmbedding(EMB):
    def __init__(self, DATA_PATH, attribute, output_path, embedding_size=2):
        super(smilessEmbedding, self).__init__(DATA_PATH, attribute, output_path)
        self.embedding_size = embedding_size
        self.mcw_dict = read_pkl('./sembed_dict.pkl')


    def clean_step(self, idx):
        stxt = self.content[idx][1:-1]
        print([i.strip()[1:-1] for i in stxt.split(',')])
        return [i.strip()[1:-1] for i in stxt.split(',')]

    def embed_step(self, clean_data):
        return self.mcv(clean_data)
    
    def mcv(self, data):
        sembs = []
        for d in data:
            sembs.append(self.mcw_dict[d])
        print(sembs)
        return sembs

    def save(self, clean_data, idx, path):
        clean_data_dict = {}
        clean_data_dict['label'] = self.label[idx]
        clean_data_dict['nctid'] = self.nctids[idx]
        clean_data_dict['smiless'] = clean_data
        save_pkl(clean_data_dict, path + self.nctids[idx] + '.pkl')
        return
    
if __name__ == '__main__':
    mcw_dict = read_pkl('./sembed_dict.pkl')
    # print(mcw_dict['[H][N]([H])([H])[Pt]1(OCC(=O)O1)[N]([H])([H])[H]'])
    # print(mcw_dict['[H][N]1([H])[C@@H]2CCCC[C@H]2[N]([H])([H])[Pt]11OC(=O)C(=O)O1'])
    se = smilessEmbedding(DATA_PATH, 'smiless', '../data/')
    se.worker()
    # ie.clean_step(1)
