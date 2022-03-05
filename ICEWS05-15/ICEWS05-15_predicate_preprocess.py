import numpy as np
import os
import pickle
import torch
import pandas as pd
from collections import defaultdict as ddict

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            #ent_set.add(head)
            #rel_set.add(rel)
            #ent_set.add(tail)
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim, split):
    e1 = [quad[0] for quad in data if quad[3] == tim] # subject in this ts
    rel = [quad[1] for quad in data if quad[3] == tim] # relation in this ts
    e2 = [quad[2] for quad in data if quad[3] == tim] # object in this ts

    triplet = np.array([e1, rel, e2]).transpose()
    triplet_unique = np.unique(triplet, axis=0) # data without inv rel

    adj_mtx_idx = [] # adjacency matrix element index whose value is 1
    sr2o = ddict(set)
    neib = ddict(set)
    so2r = ddict(set)
    for trp_idx in range(len(e1)):
        sr2o[(e1[trp_idx], rel[trp_idx])].add(e2[trp_idx])
        sr2o[(e2[trp_idx], rel[trp_idx] + num_rel)].add(e1[trp_idx])
        neib[e1[trp_idx]].add(e2[trp_idx])
        neib[e2[trp_idx]].add(e1[trp_idx])
        #adj_mtx[e1[trp_idx], e2[trp_idx]] = 1 # adjacency matrix
        #adj_mtx[e2[trp_idx], e1[trp_idx]] = 1
        adj_mtx_idx.append([e1[trp_idx], rel[trp_idx], e2[trp_idx]])
        adj_mtx_idx.append([e2[trp_idx], rel[trp_idx] + num_rel, e1[trp_idx]])
        so2r[(e1[trp_idx], e2[trp_idx])].add(rel[trp_idx])
        so2r[(e2[trp_idx], e1[trp_idx])].add(rel[trp_idx] + num_rel)
    sr2o_tmp = {k: list(v) for k, v in sr2o.items()}
    neib_tmp = {k: list(v) for k, v in neib.items()}
    so2r_tmp = {k: list(v) for k, v in so2r.items()}

    adj_mtx_idx_unique = np.unique(adj_mtx_idx, axis=0)
    adj_mtx_idx = torch.tensor(adj_mtx_idx_unique, dtype=int).t()
    adj_one = torch.ones((adj_mtx_idx.shape[1],))

    trp = [] # as input for the model
    trp_eval = [] # for evaluation
    if split == 'train':
        for (sub, pre), obj in sr2o_tmp.items():
            trp.extend([{'triple':(sub, pre, o), 'label': sr2o_tmp[(sub, pre)], 'sub_samp': 1} for o in obj])
    else:
        trp1 = []
        trp2 = []
        for trp_idx in range(triplet_unique.shape[0]):
            sub, pre, obj = triplet_unique[trp_idx,:]
            trp.append({'triple':(sub, pre, obj), 'label': sr2o_tmp[(sub, pre)], 'sub_samp': 1})
            trp.append({'triple': (obj, pre + num_rel, sub), 'label': sr2o_tmp[(obj, pre + num_rel)], 'sub_samp': 1})
            trp1.append({'triple': (sub, pre, obj), 'label': sr2o_tmp[(sub, pre)]})
            trp2.append({'triple': (obj, pre + num_rel, sub), 'label': sr2o_tmp[(obj, pre + num_rel)]})
        trp_eval = [trp1, trp2]

    return triplet_unique.transpose(), sr2o_tmp, trp, trp_eval, neib_tmp, torch.sparse_coo_tensor(adj_mtx_idx, adj_one, [num_e, 2 * num_rel, num_e]), so2r_tmp

def construct_adj(data, num_rel):
    edge_index, edge_type = [], []

    # Adding edges
    for trp_idx in range(data.shape[0]):
        sub, rel, obj = data[trp_idx,:]
        edge_index.append((sub, obj))
        edge_type.append(rel)

    # Adding inverse edges
    for trp_idx in range(data.shape[0]):
        sub, rel, obj = data[trp_idx, :]
        edge_index.append((obj, sub))
        edge_type.append(rel + num_rel)

    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)

    return edge_index, edge_type

def load_static(num_rel):
    #data = []
    sr2o = ddict(set)

    for split in ['train', 'valid', 'test']:
        for line in open('{}.txt'.format(split)):
            sub, rel, obj, _ = map(str.lower, line.strip().split('\t'))
            sub, rel, obj = int(sub), int(rel), int(obj)
            #data.append((sub, rel, obj))

            sr2o[(sub, rel)].add(obj)
            sr2o[(obj, rel + num_rel)].add(sub)

    sr2o_all = {k: list(v) for k, v in sr2o.items()}

    return sr2o_all


data = ddict(list)
sr2o_all = ddict(list)
triples = ddict(list)
adjs = ddict(list)
timestamp = ddict(list)
nei = ddict(list)
so2r_all = ddict(list)
adjlist = []
num_e, num_rel = get_total_number('', 'stat.txt')

t_indep_trp = load_static(num_rel)

for split in ['train', 'valid', 'test']:
    quadruple, ts = load_quadruples('', '{}.txt'.format(split))
    for ts_ in ts:
        print(ts_)
        timestamp[split].append(ts_)
        data_ts_, sr2o, trp, trp_eval, neib, adj_mtx, so2r = get_data_with_t(quadruple, ts_, split)
        data[split].append(data_ts_) # data without inv rel
        sr2o_all[split].append(sr2o) # with inv rel
        so2r_all[split].append(so2r)
        nei[split].append(neib)
        adjlist.append(adj_mtx)

        edge_info = ddict(torch.Tensor)
        edge_info['edge_index'], edge_info['edge_type'] = construct_adj(data_ts_.transpose(), num_rel)
        edge_info = dict(edge_info)
        adjs[split].append(edge_info)

        if split =='train':
            triples[split].append(trp) # with inv rel
        else:
            triples[split].append(trp)
            triples['{}_{}'.format(split, 'tail')].append(trp_eval[0])
            triples['{}_{}'.format(split, 'head')].append(trp_eval[1])


data = dict(data)
sr2o_all = dict(sr2o_all)
triples = dict(triples)
adjs = dict(adjs)
timestamp = dict(timestamp)
nei = dict(nei)
so2r_all = ddict(list)

with open('t_indep_trp.pkl', 'wb') as fp:
    pickle.dump(t_indep_trp, fp)

with open('data_tKG.pkl', 'wb') as fp:
    pickle.dump(data, fp)

with open('sr2o_all_tKG.pkl', 'wb') as fp:
    pickle.dump(sr2o_all, fp)

with open('triples_tKG.pkl', 'wb') as fp:
    pickle.dump(triples, fp)

with open('adjs_tKG.pkl', 'wb') as fp:
    pickle.dump(adjs, fp)

with open('timestamp_tKG.pkl', 'wb') as fp:
    pickle.dump(timestamp, fp)

with open('neighbor_tKG.pkl', 'wb') as fp:
    pickle.dump(nei, fp)

with open('adjlist_tKG.pkl', 'wb') as fp:
    pickle.dump(adjlist, fp)
    
with open('so2r_all_tKG.pkl', 'wb') as fp:
    pickle.dump(so2r_all, fp)
