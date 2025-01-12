import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
from ctp.dataset import ClinicalTtrialsPredictionDatasetH
from utils.utils import findcriterion


parser = argparse.ArgumentParser(
        prog='ModelTest',
        description='ModelTest',
    )


DATA_FOLDER = os.environ.get('DATA_FOLDER', '/home/yzhang37/Trials/data/')
# dataset
parser.add_argument('--phase', default='I', type=str, help='phase', choices=['I', 'II', 'III'])
parser.add_argument('--device', default='cuda', help='device', choices=['cuda', 'cpu'])
parser.add_argument('--batch_size', default=10, help='batch_size', type=int)

# models
parser.add_argument('--itoken_size', default=64, help='icdcodes', type=int)
parser.add_argument('--stoken_size', default=15, help='smiless', type=int)
parser.add_argument('--ctoken_size', default=768, help='criteria', type=int)
parser.add_argument('--dropout', default=0.005, help='dropout', type=float)
parser.add_argument('--nhead', default=2, help='smiless', type=int)
parser.add_argument('--nlayer', default=2, help='smiless', type=int)
parser.add_argument('--emb_size', default=8, help='smiless', type=int)
parser.add_argument('--epsilon', default=0.5, help='smiless', type=float)
parser.add_argument('--temperature', default=0.2, help='smiless', type=float)
parser.add_argument('--rho2', default=1e-3, help='smiless', type=float)
parser.add_argument('--rho1', default=1e-3, help='smiless', type=float)
parser.add_argument('--threshold', default=0.5, help='smiless', type=float)

parser.add_argument("--weighted", help="weighted loss", action="store_true")

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_length):
        super(PositionalEncoding, self).__init__()
        if emb_size % 2 != 0:
            emb_size += 1
        pos_encoding = torch.zeros(max_length, emb_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        self.pos_encoding = pos_encoding.unsqueeze(0)

    def forward(self, embeddings):
        return embeddings + Variable(self.pos_encoding[:, :embeddings.size(1), :embeddings.size(2)], requires_grad=False).to(embeddings.get_device())
    
    
class SelfAttention(nn.Module):
    def __init__(self, emb_size, nhead, dropout):
        super(SelfAttention, self).__init__()
        self.self_att = nn.MultiheadAttention(emb_size, nhead, dropout=dropout, batch_first=True)
        self.droupout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)
    
    def forward(self, tokens, masks):
        _x = tokens
        x = self.self_att(tokens, tokens, tokens, key_padding_mask=masks)[0]
        return self.norm(_x + self.droupout(x))

class CrossAttention(nn.Module):
    def __init__(self, emb_size, nhead, dropout):
        super(CrossAttention, self).__init__()
        self.cross_att = nn.MultiheadAttention(emb_size, nhead, dropout=dropout, batch_first=True)
        self.droupout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)
    
    def forward(self, quries, mem, masks):
        _x = quries
        x = self.cross_att(quries, mem, mem, key_padding_mask=masks)[0]
        return self.norm(_x + self.droupout(x))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, emb_size, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_size, int(emb_size / 2))
        self.linear2 = nn.Linear(int(emb_size / 2), emb_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        _x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.norm(_x + self.dropout2(x))
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, emb_size, nhead, dropout):
        super(SelfAttentionLayer, self).__init__()
        self.self_att = SelfAttention(emb_size, nhead, dropout)
        self.ffn = PositionwiseFeedForward(emb_size, dropout)
    
    def forward(self, tokens, masks):
        x = self.self_att(tokens, masks)
        return self.ffn(x)
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, emb_size, nhead, dropout):
        super(CrossAttentionLayer, self).__init__()
        self.cross_att = CrossAttention(emb_size, nhead, dropout)
        self.ffn = PositionwiseFeedForward(emb_size, dropout)
    
    def forward(self, quries, mem, masks):
        x = self.cross_att(quries, mem, masks)
        return self.ffn(x)


class ResidualProjection(nn.Module):
    def __init__(self, token_size, emb_size):
        super(ResidualProjection, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(token_size, int((emb_size + token_size) / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int((emb_size + token_size) / 2), emb_size),
        )
        if token_size != emb_size:
            self.shortcut = nn.Linear(token_size, emb_size)
        else:
            self.shortcut = nn.Identity()

    def forward(self, token):
        return self.main(token)  + self.shortcut(token)

class Router(nn.Module):
    def __init__(self, emb_size):
        super(Router, self).__init__()
        self.fc = ResidualProjection(emb_size, 1)
    
    def forward(self, tokens):
        return F.sigmoid(self.fc(tokens))
    
def remasks(masks, prob, threshold):
    if threshold <= 0 or threshold >= 1:
        return masks
    else:
        # print('kk', prob.squeeze() < threshold)
        return masks | (prob < threshold).squeeze()

class CauchyLoss(nn.Module):
    def __init__(self, epsilon):
        super(CauchyLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, prob):
        return torch.log(1 + (prob / self.epsilon)**2).mean()
    
def seq_similarity(seq1, seq2, masks1, masks2):

    masked_seq1 = seq1 * (~masks1).unsqueeze(-1)
    masked_seq2 = seq2 * (~masks2).unsqueeze(-1)
    indices1 = torch.randperm(masked_seq1.size(1))
    indices2 = torch.randperm(masked_seq2.size(1))
    shuffled_seq1 = masked_seq1[:, indices1, :]
    shuffled_seq2 = masked_seq2[:, indices2, :]

    len_seq = min(shuffled_seq1.size(1), shuffled_seq2.size(1))
    # print(seq2_masked.size())
    # print(seq1_masked.size())
    # print(len_seq)
    

    return F.cosine_similarity(shuffled_seq1[:, :len_seq, :], shuffled_seq2[:, :len_seq, :], dim=-1)

    
class NPairLoss(nn.Module):
    def __init__(self, temperature):
        super(NPairLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, qis, qcs, qsi, qci, qsc, qic, mis, mcs, msi, mci, msc, mic):
        sim_pos = seq_similarity(qis, qsi, mis, msi) + seq_similarity(qcs, qsc, mcs, msc) + seq_similarity(qci, qic, mci, mic)
        sim_neg = seq_similarity(qis, qci, mis, mci) + seq_similarity(qis, qsc, mis, msc) + seq_similarity(qsi, qic, msi, mic) + seq_similarity(qsi, qcs, msi, mcs) + seq_similarity(qcs, qic, mcs, mic) + seq_similarity(qsc, qci, msc, mci)

        # print('pos', sim_pos)
        # print('neg', sim_neg)

        log_prob_pos = F.log_softmax(sim_pos / self.temperature, dim=-1)
        log_prob_neg = F.log_softmax(sim_neg / self.temperature, dim=-1)

        return -(log_prob_pos+log_prob_neg).mean()




class ClinicalTtrialsPredictionModelH_nlayers(nn.Module):
    def __init__(self, args, max_length={'icds':5, 'smiless':5, 'in_criteria':5, 'ex_criteria':3}):
        super(ClinicalTtrialsPredictionModelH_nlayers, self).__init__()
        self.threshold = args.threshold
        self.sprojection = ResidualProjection(args.stoken_size, args.emb_size)
        self.s_pe = PositionalEncoding(args.emb_size, max_length['smiless'])
        self.cprojection = ResidualProjection(args.ctoken_size, args.emb_size)
        self.inc_pe = PositionalEncoding(args.emb_size, max_length['in_criteria'])
        self.exc_pe = PositionalEncoding(args.emb_size, max_length['ex_criteria'])
        self.iprojection = ResidualProjection(args.itoken_size, args.emb_size)
        self.i_pe = PositionalEncoding(args.emb_size, max_length['icds'])
        
        NAMELIST= {
            'selfatt': ['i', 's', 'c'],
            'crossatt': ['is', 'ic', 'si', 'sc', 'ci', 'cs'],
            'router': ['is', 'ic', 'si', 'sc', 'inci', 'incs', 'exci', 'excs'],
        }

        self.self_att_layers = {}
        self.cross_att_layers = {}
        self.routers = {}

        for selfatt in NAMELIST['selfatt']:
            self.self_att_layers[selfatt] = nn.ModuleList([SelfAttentionLayer(args.emb_size, args.nhead, args.dropout) for _ in range(args.nlayer)]).to(args.device)

        for crossatt in NAMELIST['crossatt']:
            self.cross_att_layers[crossatt] = nn.ModuleList([CrossAttentionLayer(args.emb_size, args.nhead, args.dropout) for _ in range(args.nlayer)]).to(args.device)
        
        for router in NAMELIST['router']:
            self.routers[router] = Router(args.emb_size).to(args.device)

        self.all_pe = PositionalEncoding(args.emb_size, 2*(max_length['icds']+max_length['smiless']+max_length['in_criteria']+max_length['ex_criteria']))
        self.compensation = nn.ModuleList([SelfAttentionLayer(args.emb_size, args.nhead, args.dropout) for _ in range(args.nlayer)])
        self.predictionhead = ResidualProjection(args.emb_size, 1)

        self.rho1 = args.rho1
        self.rho2 = args.rho2
        self.cauchyloss = CauchyLoss(args.epsilon)
        self.ntxloss = NPairLoss(args.temperature)
        self.bceloss = findcriterion(args.weighted, args.phase, args.device)
    

    
    def forward(self, itokens, imasks, in_ctokens, in_cmasks, ex_ctokens, ex_cmasks, stokens, smasks, labels):
        s = self.sprojection(stokens)
        s = self.s_pe(s)
        for self_att_layer in self.self_att_layers['s']:
            s = self_att_layer(s, smasks)
        psi = self.routers['si'](s)
        psc = self.routers['sc'](s)
        msi = remasks(smasks, psi, self.threshold)
        msc = remasks(smasks, psc, self.threshold)
        cpsi = self.cauchyloss(psi)
        cpsc = self.cauchyloss(psc)
        


        i = self.iprojection(itokens)
        i = self.i_pe(i)
        for self_att_layer in self.self_att_layers['i']:
            i = self_att_layer(i, imasks)
        pis = self.routers['is'](i)
        pic  = self.routers['ic'](i)
        mis = remasks(imasks, pis, self.threshold)
        mic = remasks(imasks, pic, self.threshold)
        cpis = self.cauchyloss(pis)
        cpic = self.cauchyloss(pic)

        in_c = self.cprojection(in_ctokens)
        in_c = self.inc_pe(in_c)
        for self_att_layer in self.self_att_layers['c']:
            in_c = self_att_layer(in_c, in_cmasks)
        pincs = self.routers['incs'](in_c)
        pinci = self.routers['inci'](in_c)
        mincs = remasks(in_cmasks, pincs, self.threshold)
        minci = remasks(in_cmasks, pinci, self.threshold)

        ex_c = self.cprojection(ex_ctokens)
        ex_c = self.exc_pe(ex_c)
        for self_att_layer in self.self_att_layers['c']:
            ex_c = self_att_layer(ex_c, ex_cmasks)
        pexcs = self.routers['excs'](ex_c)
        pexci = self.routers['exci'](ex_c)
        mexcs = remasks(ex_cmasks, pexcs, self.threshold)
        mexci = remasks(ex_cmasks, pexci, self.threshold)

        c = torch.cat((in_c, ex_c), axis=1)

        cmasks = torch.cat((in_cmasks, ex_cmasks), axis=1)
        pcs = torch.cat((pincs, pexcs), axis=1)
        pci = torch.cat((pinci, pexci), axis=1)
        mcs = torch.cat((mincs, mexcs), axis=1)
        mci = torch.cat((minci, mexci), axis=1)
        cpcs = self.cauchyloss(pcs)
        cpci = self.cauchyloss(pci)

        qis = pis * i
        for cross_att_layer in self.cross_att_layers['is']:
            qis = cross_att_layer(qis, s, smasks)
        qcs = pcs * c
        for cross_att_layer in self.cross_att_layers['cs']:
            qcs = cross_att_layer(qcs, s, smasks)

        qsi = psi * s
        for cross_att_layer in self.cross_att_layers['si']:
            qsi = cross_att_layer(qsi, i, imasks)
        qci = pci * c
        for cross_att_layer in self.cross_att_layers['ci']:
            qci = cross_att_layer(qci, i, imasks)

        qic = pic * i
        for cross_att_layer in self.cross_att_layers['ic']:
            qic = cross_att_layer(qic, c, cmasks)
        qsc = psc * s
        for cross_att_layer in self.cross_att_layers['sc']:
            qsc = cross_att_layer(qsc, c, cmasks)
        # print(qsc.size())

        all_tokens = torch.cat((qis, qcs, qsi, qci, qic, qsc), axis=1)
        # print(all_tokens.size())
        all_masks = torch.cat((mis, mcs, msi, mci, mic, msc), axis=1)
        # print('mask', all_masks.size())
        all_tokens = self.all_pe(all_tokens)
        # print(all_tokens.size())
        for self_att_layer in self.compensation:
            all_tokens = self_att_layer(all_tokens, all_masks)
        # print(all_tokens)
        # print(~all_masks)
        # masked_sum = torch.mean(all_tokens * (~all_masks).unsqueeze(-1), dim=1)
        masked_sum = torch.sum(all_tokens * (~all_masks).unsqueeze(-1), dim=1)
        cauchyloss = cpci + cpcs + cpis + cpsi + cpsc + cpic
        ntxloss = self.ntxloss(qis, qcs, qsi, qci, qsc, qic, mis, mcs, msi, mci, msc, mic)
        preds = self.predictionhead(masked_sum)
        bceloss = self.bceloss(preds, labels.float().view(-1, 1))
        return {
            'preds':preds,
            'cauchyloss': cauchyloss,
            'ntxloss': ntxloss,
            'bceloss': bceloss,
            'loss': bceloss + self.rho1 * cauchyloss + self.rho2 * ntxloss

        }
        # print(masked_sum)
        # print(masked_sum.size())
if __name__ == '__main__':

    args = parser.parse_args()
    model = ClinicalTtrialsPredictionModelH_nlayers(args)
    model.to(args.device)


    dataset = ClinicalTtrialsPredictionDatasetH(DATA_FOLDER, 'train', 'III')
    dataloader = DataLoader(dataset , batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    c = 0
    for d in dataloader:
        # print(d['itokens'].size())
        # print(d['imasks'].size())
        # print(d['stokens'].size())
        # print(d['in_ctokens'].size())
        # print(d['ex_ctokens'].size())
        # print('-------------------------')
        itokens = d['itokens'].to(args.device)
        imasks = d['imasks'].to(args.device)
        stokens = d['stokens'].to(args.device)
        smasks = d['smasks'].to(args.device)
        in_ctokens = d['in_ctokens'].to(args.device)
        in_cmasks = d['in_cmasks'].to(args.device)
        ex_ctokens = d['ex_ctokens'].to(args.device)
        ex_cmasks = d['ex_cmasks'].to(args.device)
        label = d['label'].to(args.device)
        # print(d['itokens'].size())
        # print(d['in_ctokens'].size())
        # print(d['ex_ctokens'].size())
        # print(d['stokens'].size())
        # print(d['nctid'])
        # print(d['ex_cmasks'])
        # print(d['in_cmasks'])
        # print(label)
        outputs = model(itokens, imasks, in_ctokens, in_cmasks, ex_ctokens, ex_cmasks, stokens, smasks, label)
        print(outputs)
        # print(outputs)
        # loss = criterion(outputs, label.float().view(-1, 1))
        optimizer.zero_grad()
        outputs['loss'].backward()
        optimizer.step()
        # print(loss)
        # # print('gcls', model.globalTokenEmbedding.cls_token)
        # # print('gpe', model.globalTokenEmbedding.pos_embedding)
        c += 1
        # # print(d['label'])
