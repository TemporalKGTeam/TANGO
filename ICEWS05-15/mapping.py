import numpy as np
import pandas as pd
from ordered_set import OrderedSet

file1 = pd.read_table("05-15train.txt", sep='\t')
file2 = pd.read_table("05-15valid.txt", sep='\t')
file3 = pd.read_table("05-15test.txt", sep='\t')
file = pd.concat([file1,file2,file3])
file = file.drop_duplicates()
nodes = np.unique(file['sub'].to_list() + file['obj'].to_list())
print("number of entities:", len(nodes))
print("number of interactions:", file.shape[0])

len_train, len_val = int(file.shape[0] * 0.8)+41, int(file.shape[0] * 0.1)+56
len_test = file.shape[0] - len_train - len_val
time_s = file['time'].to_list()
print(len(np.unique(time_s)))
print(np.unique((time_s))[0], np.unique((time_s))[-1])
print(file)

quadruples = file.sort_values(by="time" , ascending=True)
print(quadruples)
ent_set, rel_set, t_set = OrderedSet(), OrderedSet(), OrderedSet()
for quad in quadruples.itertuples():
    sub, rel, obj, t = getattr(quad, 'sub'), getattr(quad, 'rel'), getattr(quad, 'obj'), getattr(quad, 'time')
    ent_set.add(sub)
    rel_set.add(rel)
    ent_set.add(obj)
    t_set.add(t)

ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
rel2id.update({rel+'_reverse': idx+len(rel2id) for idx, rel in enumerate(rel_set)})
t2id = {t: idx*24 for idx, t in enumerate(t_set)}

id2ent = {idx: ent for ent, idx in ent2id.items()}
id2rel = {idx: rel for rel, idx in rel2id.items()}
id2t = {idx: t for t, idx in t2id.items()}

stat = open("stat.txt", "w")
stat.write(str(len(ent2id)))
stat.write('\t')
stat.write(str(len(rel2id)//2))
stat.close()

train_quad, val_quad, test_quad = quadruples.iloc[0:len_train], quadruples.iloc[len_train:len_train+len_val], quadruples[len_train+len_val:]
tr = open('train.txt','w')
for quad_tr in train_quad.itertuples():
    sub, rel, obj, t = getattr(quad_tr, 'sub'), getattr(quad_tr, 'rel'), getattr(quad_tr, 'obj'), getattr(quad_tr, 'time')
    tr.write(str(ent2id[sub]))
    tr.write('\t')
    tr.write(str(rel2id[rel]))
    tr.write('\t')
    tr.write(str(ent2id[obj]))
    tr.write('\t')
    tr.write(str(t2id[t]))
    tr.write('\n')
tr.close()

val = open('valid.txt','w')
for quad_val in val_quad.itertuples():
    sub, rel, obj, t = getattr(quad_val, 'sub'), getattr(quad_val, 'rel'), getattr(quad_val, 'obj'), getattr(quad_val, 'time')
    val.write(str(ent2id[sub]))
    val.write('\t')
    val.write(str(rel2id[rel]))
    val.write('\t')
    val.write(str(ent2id[obj]))
    val.write('\t')
    val.write(str(t2id[t]))
    val.write('\n')
val.close()

te = open('test.txt','w')
for quad_te in test_quad.itertuples():
    sub, rel, obj, t = getattr(quad_te, 'sub'), getattr(quad_te, 'rel'), getattr(quad_te, 'obj'), getattr(quad_te, 'time')
    te.write(str(ent2id[sub]))
    te.write('\t')
    te.write(str(rel2id[rel]))
    te.write('\t')
    te.write(str(ent2id[obj]))
    te.write('\t')
    te.write(str(t2id[t]))
    te.write('\n')
te.close()