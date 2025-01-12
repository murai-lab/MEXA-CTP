import pandas as pd
from tqdm import tqdm
from glob import glob

import sys
from utils.utils_whole import path_exist, create_path, read_pkl, save_pkl


class Embedding(object):
    def __init__(self, DATA_PATH, attribute, output_path):
        self.DATA_PATH = DATA_PATH

        if attribute not in ['icdcodes', 'smiless', 'criteria']:
            raise NotImplemented
        
        self.attribute = attribute
        self.preprocess_path = output_path + self.attribute + '/preprocess/'
        self.embedding_path = output_path + self.attribute + '/embed/'
        
        create_path(self.preprocess_path)
        create_path(self.embedding_path)

        data = pd.read_csv(self.DATA_PATH)

        self.nctids = data['nctid']
        self.label = data['label']
        self.content = data[self.attribute]


    def clean_step(self,):
        pass


    def embed_step(self,):
        pass

    def save(self, data, idx, path):
        pass

    def worker(self,):
        emb_path = glob(f'{self.embedding_path}*.pkl')
        print(len(emb_path))
        for idx, nctid in tqdm(enumerate(self.nctids)):   
            if f'{self.embedding_path}{nctid}.pkl' in emb_path:
                continue
            clean_data = self.clean_step(idx)
            # print(len(clean_data))
            self.save(clean_data, idx, self.preprocess_path)
            emb = self.embed_step(clean_data)
            # print('SIZE')
            # print(emb[0].size())
            # print(emb[1].size())
            self.save(emb, idx, self.embedding_path)
        return
