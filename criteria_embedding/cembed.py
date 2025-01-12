import os
import torch

import sys
sys.path.insert(0, '/home/yzhang37/Trials/')
from utils.embedding import Embedding as EMB
from utils.utils_whole import save_pkl

DATA_PATH = os.environ.get('DATA_PATH', '/home/yzhang37/data/clinicalTrialsPrediction/clinical-trial-outcome-prediction/data/raw_data.csv')


class criteriaEmbedding(EMB):
    def __init__(self, DATA_PATH, attribute, output_path, max_length=32):
        super(criteriaEmbedding, self).__init__(DATA_PATH, attribute, output_path)
        self.embedding_size = 768 # biobert
        self.max_length = max_length

    def split(self, criteria):
        criteria_split = criteria.lower().split('\n')
        
        # filter
        filter_fn = lambda x: len(x.strip()) > 0
        strip_fn = lambda x:x.strip()

        criteria_split = list(filter(filter_fn, criteria_split))
        criteria_split = list(map(strip_fn, criteria_split))
        return criteria_split

    def find_pattern(self, criteria_split, idx):
        if '-  inclusion criteria' in criteria_split[0]:
            return self.pattern_1(criteria_split)
        
        elif 'inclusion criteria' in criteria_split[0]:
            return self.pattern_2(criteria_split)
        
        elif '-  conditions for patient eligibility' in criteria_split[0]:
            return self.pattern_1(criteria_split)

        elif 'conditions for patient eligibility' in criteria_split[0]:
            return self.pattern_2(criteria_split)
        
        elif 'disease characteristics' in criteria_split[0]:
            return self.pattern_3(criteria_split)
        elif '1.' in criteria_split[0]:
            return self.pattern_4(criteria_split)
        else:
            print(self.nctids[idx])
            return self.pattern_5(criteria_split)
        
    def pattern_1(self, criteria_split):
        clean_data = []
        c_ls = []
        for cs in criteria_split:
            if '-  inclusion criteria' in cs or '-  exclusion criteria' in cs or '-  conditions for patient eligibility' in cs or '-  conditions for patient ineligibility' in cs:
                if len(c_ls) > 0:
                    clean_data.append(c_ls)
                c_ls = []
                in_c = ''
            else:
                in_c += cs
                if cs[-1] == '.':
                    if len(in_c) > 0:
                        c_ls.append(in_c)
                    in_c = ''
        if len(c_ls) > 0:
            clean_data.append(c_ls)
        return clean_data
    
    def pattern_2(self, criteria_split):
        clean_data = []
        c_ls = []
        for cs in criteria_split:
            if 'inclusion criteria' in cs or 'exclusion criteria' in cs or 'conditions for patient eligibility' in cs or 'conditions for patient ineligibility' in cs:
                if len(c_ls) > 0:
                    clean_data.append(c_ls)
                c_ls = []
                in_c = ''
            else:
                if '-  ' in cs:
                    if len(in_c) > 0:
                        c_ls.append(in_c)
                    in_c = cs[3:]
                else:
                    in_c += ' ' +  cs
        if len(c_ls) > 0:
            clean_data.append(c_ls)
        return clean_data


    def pattern_3(self,criteria_split):
        clean_data = []
        c_ls = []
        in_c = ''
        for cs in criteria_split:
            if 'prior concurrent therapy:' in cs:
                if len(in_c) > 0:
                    c_ls.append(in_c)
                break
            if 'patient characteristics:' in cs:
                in_c = cs.replace('patient characteristics:', '')
            else:
                if len(in_c) != 0:
                    in_c += ' ' + cs
        
        if len(c_ls) > 0:
            clean_data.append(c_ls)
        return clean_data
    def pattern_4(self, criteria_split):
        clean_data = []
        c_ls = []
        in_c = ''
        id = 1
        for cs in criteria_split:
            if f'{id}.' in cs:
                if len(in_c) > 0:
                    c_ls.append(in_c)
                in_c = cs.replace(f'{id}. ', '')
                id += 1
            else:
                in_c += ' ' + cs
        if len(in_c) > 0:
            c_ls.append(in_c)
        if len(c_ls) > 0:
            clean_data.append(c_ls)
        return clean_data


    def pattern_5(self, criteria_split):
        clean_data = []
        c_ls = []
        in_c = ''
        for cs in criteria_split:
            in_c += cs
        c_ls.append(in_c)
        clean_data.append(c_ls)
        return clean_data

    def clean_step(self, idx):
        if isinstance(self.content[idx], float): # NUll value
            return []
        criteria_split = self.split(self.content[idx])
        clean_data =  self.find_pattern(criteria_split, idx)
        return clean_data
        print(self.nctids[idx])
        print(criteria_split)
        print('*********')
        print(self.find_pattern(criteria_split))
        return
    
    def biobert(self, data):
        # Load model directly
        print(data)
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        encoded_texts = tokenizer(data, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_attention_mask=True)
        input_ids = torch.tensor(encoded_texts['input_ids'])
        attention_mask = torch.tensor(encoded_texts['attention_mask'])
        print(input_ids)
        #  inputs = tokenizer(data, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)[0]
            print(outputs.size())
            return [outputs, attention_mask]

    def embed_step(self, clean_data):
        print(len(clean_data))
        if len(clean_data) == 0:
            return [torch.zeros(1, self.max_length, 768), torch.zeros(1, self.max_length, 768)]
        elif len(clean_data) == 1:
            return [self.biobert(clean_data[0]), torch.zeros(1, self.max_length, 768)]
        else:
            # print(len(clean_data[0]))
            return [self.biobert(clean_data[0]), self.biobert(clean_data[1])]
    
    def save(self, clean_data, idx, path):
        clean_data_dict = {}
        clean_data_dict['label'] = self.label[idx]
        clean_data_dict['nctid'] = self.nctids[idx]
        clean_data_dict['criteria'] = clean_data
        save_pkl(clean_data_dict, path + self.nctids[idx] + '.pkl')
        return


if __name__ == '__main__':
    ce = criteriaEmbedding(DATA_PATH, 'criteria', '../data/')
    ce.worker()
    # ce.clean_step(1)


    