#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AdamW, WarmupLinearSchedule

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm_notebook as tqdm
import csv
import random
import matplotlib.pyplot as plt
import sys


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("using cuda!")
device = torch.device("cuda:"+sys.argv[2])
col_names = []
raw_data = []

with open("data/task2_trainset.csv") as f:
    rows = csv.reader(f)
    for row in rows:
        if row[0] == "Id":
            col_names = row
            continue
        raw_data.append(row)

categories = {"THEORETICAL": 0, "ENGINEERING": 1, "EMPIRICAL": 2, "OTHERS": 3}

for row in raw_data:
    row[2] = row[2].replace("$$$", " ")
    row[3] = row[3].split("/")
    row[4] = row[4].split("/")
    row[6] = row[6].split()
    tmp = [0, 0, 0, 0]
    for cat in row[6]:
        tmp[categories[cat]] = 1
    row.remove(row[6])
    if tmp == [1, 0, 0, 0]:
        tmp = [0] + tmp
    elif tmp == [0, 1, 0, 0]:
        tmp = [1] + tmp
    elif tmp == [0, 0, 1, 0]:
        tmp = [2] + tmp
    elif tmp == [1, 1, 0, 0]:
        tmp = [3] + tmp
    elif tmp == [1, 0, 1, 0]:
        tmp = [4] + tmp
    elif tmp == [0, 1, 1, 0]:
        tmp = [5] + tmp
    elif tmp == [1, 1, 1, 0]:
        tmp = [6] + tmp
    else:
        tmp = [7] + tmp
    assert len(tmp) == 5
    row += tmp
    
ratios = [0.0, 0.0, 0.0, 0.0]
base = [0.0, 0.0, 0.0, 0.0]

for i in range(-4, 0):
    ratios[i] = sum([x[i] for x in raw_data]) / len(raw_data)
    base[i] = 2*ratios[i] / (1+ratios[i])
    
ratios, base


# In[4]:


class Data():
    
    def __init__(self, idxs, is_test):
        
        self.title = []
        self.sent = []
        self.label = []
        self.category = []
        self.idxs = idxs
        self.is_test = is_test
    
    def readData(self, data_path):
        
        with open(data_path) as fd:
            rows = csv.reader(fd)
            first = True
            idx = 0
            for row in rows:
                if first:
                    first = False
                    continue
                elif idx in self.idxs:
                    self.title.append(row[1])
                    self.sent.append(row[2].replace("$$$", " "))
#                     self.category.append(row[4].split("/"))
#                     self.category[-1] = [cat.split(".")[0] for cat in self.category[-1]]
                    self.category.append([0 for _ in range(len(list(domains.keys())))])
                    for cate in row[4].split("/"):
                        self.category[-1][domains[cate.split(".")[0]]] = 1
                    if not self.is_test:
                        cates = row[6].split()
                        tmp = [0, 0, 0, 0]
                        for cate in cates:
                            tmp[categories[cate]] = 1
                        self.label.append(tmp)
                idx += 1
        
    def batchData(self, batch_size):
        
        batch_num = np.ceil(len(self.sent) / batch_size)
        batch_X = []
        batch_Y = []
        
        for b in range(batch_num):
            batch_X.append(self.sent[b*batch_size:(b+1)*batch_size])
            batch_Y.append(self.label[b*batch_size:(b+1)*batch_size])
        
        return batch_X, batch_Y
        


# In[5]:


def count_category(data):
    ret = {}
    for idx, cgys in enumerate(data.category):
        for cgy in cgys:
            if cgy not in list(ret.keys()):
                ret[cgy] = [0, 0, 0, 0]
            ret[cgy] = [ret[cgy][i]+data.label[idx][i] for i in range(4)]
    return ret


# In[6]:


data_dir = "data"
test_data = Data(idxs=[i for i in range(20000)], is_test=True)
test_data.readData(data_dir+"/task2_public_testset.csv")


# In[7]:


test_data.title[0], test_data.sent[0], test_data.category[0], len(test_data.sent)


# ## Fine-Tune Bert

# In[9]:


import math

class gelu(nn.Module):
    
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * cdf

    
class MutiLabelModel(nn.Module):
    
    def __init__(self, encoder, emb_size=1024, out_size=4, ce_size=23, hidden=256): # hidden=256
        super(MutiLabelModel, self).__init__()
        
        self.encoder = encoder
        self.fn_size = emb_size
        
#         self.fc1 = nn.Linear(emb_size, hidden)
#         self.rnn1 = nn.LSTM(emb_size, hidden, 1, batch_first=True, bidirectional=True) # False
#         self.rnn2 = nn.LSTM(hidden, hidden, 1, batch_first=True, bidirectional=True)
#         self.softmax = nn.Softmax(dim=1)
        self.out_fn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.fn_size, self.fn_size//2),
            gelu(),
            nn.Dropout(0.2),
            nn.Linear(self.fn_size//2, self.fn_size//8),
        )
        
    def forward(self, inp, seg_inp, cat_emb=None, cls_loc=0): # , inp_title, seg_inp_title, cls_loc=0):

        # batch = 1
        embs = self.encoder(inp, token_type_ids=seg_inp)[0] # [batch, seq, hidden]
#         embs_title = self.encoder(inp_title, token_type_ids=seg_inp_title)[0] # [batch, seq, hidden]
        outputs = embs[:, cls_loc, :]
#         outputs = torch.cat((cat_emb, outputs), 1)
#         outputs_title = embs_title[:, cls_loc, :]
#         outputs = torch.cat((outputs, outputs_title), 1)
#         embs_title = self.encoder(inp_title, token_type_ids=seg_inp_title)[0]
#         outputs, _ = self.rnn1(embs)
#         outputs_title, _ = self.rnn1(embs_title)
#         outputs = outputs.transpose(0, 1)[cls_loc]
#         outputs_title = outputs_title.transpose(0, 1)[cls_loc]
#         outputs = torch.cat((outputs, outputs_title), 1)

#         emb_title = self.encoder(inp_title, token_type_ids=seg_inp_title)[0][:, cls_loc, :] # [batch, emb]
#         emb_title = self.fc1(emb_title) # [batch, hidden]
#         outputs, _ = self.rnn1(embs) # [batch, seq, hidden]
#         attention = torch.mm(emb_title, outputs.squeeze(0).transpose(0, 1)) # [batch, seq]
#         attention = self.softmax(attention).transpose(0, 1) # [seq, batch]
#         outputs = outputs.squeeze(0) * attention # [seq, hidden]
#         outputs, _ = self.rnn2(outputs.unsqueeze(0)) # [batch, seq, hidden]
#         outputs = outputs.transpose(0, 1)[cls_loc] # [batch, hidden*2]
        outputs = self.out_fn(outputs)
#         outputs = torch.cat((cat_emb, outputs), 1)
#         outputs = self.out_fn2(outputs)
        return outputs


# In[10]:


def micro_f1_score(pred, label):
    
    TP = torch.mul(pred, label)[0]
    FP = (torch.mul(pred, (label-1)) != 0)[0]
    FN = (torch.mul(pred-1, label) != 0)[0]
    
    precision = TP.sum() / (TP.sum() + FP.sum())
    recall = TP.sum() / (TP.sum() + FN.sum())
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1, TP, FP, FN


# In[11]:


# %%time
good_seed = [5422, 6550, 6545, 9436, 5487, 1070, 7429]
take = 4
Target = -5
train = []
test = []
seed = int(sys.argv[1]) # tune this only
torch.manual_seed(seed)
best_f1 = -1
MAX_LEN = 512
pos_weight = torch.FloatTensor([1., 1., 1.75, 7.5]).to(device)

batch_size = 32
epochs = 7
lr = 1e-5

# for sss
x = [row[:Target] for row in raw_data]
y = [row[Target] for row in raw_data]

# sss
splits = StratifiedShuffleSplit(n_splits=1, random_state=seed)
for tr, te in splits.split(x, y):
    train, test = tr, te
# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# for i, (tr, te) in enumerate(folds.split(x, y)):
#     if i == 7:
#         train, test = tr, te
#         break

# read train and dev data
train_data = Data(idxs=train, is_test=False)
train_data.readData(data_dir+"/task2_trainset.csv")

dev_data = Data(idxs=test, is_test=False)
dev_data.readData(data_dir+"/task2_trainset.csv")

print("train len:{}\ttest len:{}".format(len(train_data.sent), len(dev_data.sent)))
print("train ratios:", end=" ")
for i in range(4):
    print(sum([x[i] for x in train_data.label])/len(train_data.label), end="\t")
print("")
print("dev ratios:", end="   ")
for i in range(4):
    print(sum([x[i] for x in dev_data.label])/len(dev_data.label), end="\t")
    
# count category
# category_counts = count_category(train_data)

# load pre-trained bert
print("\nloading bert...")
tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased')
encoder = BertModel.from_pretrained('scibert_scivocab_uncased')
encoder.load_state_dict(torch.load("./model/task1/encoder_{}_state".format(9487), map_location=device))

# init model
print("initialize model...")
model = MutiLabelModel(encoder, 768, take)
# model.load_state_dict(torch.load("./model/task2/model_{}_state".format(seed), map_location=device))
# print(1/0)

num_total_steps = np.ceil(len(train_data.sent) / batch_size)*epochs
num_warmup_steps = int(num_total_steps * 0.5)

optim = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = WarmupLinearSchedule(optim, warmup_steps=num_warmup_steps, t_total=num_total_steps)

if use_cuda:
    model = model.to(device)

# train and validation
print("start training!")
for ep in range(epochs):
    model = model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    thrld = np.ones((1,take))*0.5
    thrld[0][0] = 0.35
    thrld[0][1] = 0.3
    thrld[0][2] = 0.25
    thrld[0][3] = 0.35
    total_loss = 0.0

    train_metric = { "TP":np.zeros(take), "FP":np.zeros(take), "FN":np.zeros(take), "F1":[] }
    if use_cuda:
        thrld_ten = torch.from_numpy(thrld).float().to(device)
        train_metric["TP"] = torch.from_numpy(train_metric["TP"]).float().to(device)
        train_metric["FP"] = torch.from_numpy(train_metric["FP"]).float().to(device)
        train_metric["FN"] = torch.from_numpy(train_metric["FN"]).float().to(device)
    else:
        thrld_ten = torch.from_numpy(thrld).float()

    optim.zero_grad()
    pbar = enumerate(tqdm(train_data.sent))
    
    for sidx, s in pbar:

        if (sidx+1) % batch_size == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()

        # abstract
        indexed_tokens = tokenizer.add_special_tokens_single_sequence(tokenizer.encode(s))[:MAX_LEN]
        input_ids = torch.tensor([indexed_tokens]).long()
        if use_cuda:
            input_ids = input_ids.to(device)

        segments_ids = [ 0 for i in range(len(indexed_tokens)) ]
        segments_ids = torch.tensor([segments_ids]).long()
        if use_cuda:
            segments_ids = segments_ids.to(device)
            target = torch.FloatTensor(train_data.label[sidx][:take]).to(device).view(1,-1)
        else:
            target = torch.FloatTensor(train_data.label[sidx][:take]).view(1,-1)

        out = model(input_ids, segments_ids) # , category_emb)

        l = criterion(out, target)
        total_loss += l.item()

        l.backward()

        out = torch.sigmoid(out)
        pred = (out > thrld_ten.expand(target.size())).float()
        f1, tp, fp, fn = micro_f1_score(pred, target)

        train_metric["F1"].append(f1)
        train_metric["TP"] += tp.float()
        train_metric["FP"] += fp.float()
        train_metric["FN"] += fn.float()

    optim.step()
    scheduler.step()

    train_precision_all = train_metric["TP"].sum().item() / (train_metric["TP"].sum().item() + train_metric["FP"].sum().item())
    train_recall_all = train_metric["TP"].sum().item() / (train_metric["TP"].sum().item() + train_metric["FN"].sum().item())
    train_micro_f1 = (2 * train_precision_all * train_recall_all) / (train_precision_all + train_recall_all)

    avg_loss = total_loss / len(train_data.sent)

    print("Train Loss:{}\tmicro_f1:{}".format(avg_loss, train_micro_f1))
    print("micro_f1s:", end="")
    for i in range(take):
        precision = train_metric["TP"][i].item() / (train_metric["TP"][i].item()+train_metric["FP"][i].item()+1e-10)
        recall = train_metric["TP"][i].item() / (train_metric["TP"][i].item()+train_metric["FN"][i].item()+1e-10)
        print("{}".format(2*precision*recall / (precision+recall+1e-10)), end="\t")
    print("")

    # Evaluation
    model = model.eval()
    criterion = nn.BCEWithLogitsLoss()

    dev_loss = 0.0
    dev_metric = { "TP":np.zeros(take), "FP":np.zeros(take), "FN":np.zeros(take), "F1":[] }
    if use_cuda:
        dev_metric["TP"] = torch.from_numpy(dev_metric["TP"]).float().to(device)
        dev_metric["FP"] = torch.from_numpy(dev_metric["FP"]).float().to(device)
        dev_metric["FN"] = torch.from_numpy(dev_metric["FN"]).float().to(device)

    with torch.no_grad():

        for sidx, s in enumerate(tqdm(dev_data.sent)):

            # abstract
            indexed_tokens = tokenizer.add_special_tokens_single_sequence(tokenizer.encode(s))[:MAX_LEN]
            input_ids = torch.tensor([indexed_tokens]).long()
            if use_cuda:
                input_ids = input_ids.to(device)

            segments_ids = [ 0 for i in range(len(indexed_tokens)) ]
            segments_ids = torch.tensor([segments_ids]).long()
            if use_cuda:
                segments_ids = segments_ids.to(device)
                target = torch.FloatTensor(dev_data.label[sidx][:take]).to(device).view(1,-1)
            else:
                target = torch.FloatTensor(dev_data.label[sidx][:take]).view(1,-1)

            out = model(input_ids, segments_ids) # , category_emb)

            l = criterion(out, target)
            dev_loss += l

            out = torch.sigmoid(out)
            pred = (out > thrld_ten.expand(target.size())).float()
            f1, tp, fp, fn = micro_f1_score(pred, target)

            dev_metric["F1"].append(f1)
            dev_metric["TP"] += tp.float()
            dev_metric["FP"] += fp.float()
            dev_metric["FN"] += fn.float()

            if use_cuda:
                input_ids = input_ids.cpu()
                target = target.cpu()

        dev_precision_all = dev_metric["TP"].sum().item() / (dev_metric["TP"].sum().item() + dev_metric["FP"].sum().item())
        dev_recall_all = dev_metric["TP"].sum().item() / (dev_metric["TP"].sum().item() + dev_metric["FN"].sum().item())
        dev_micro_f1 = (2 * dev_precision_all * dev_recall_all) / (dev_precision_all + dev_recall_all)

        avg_loss = dev_loss / len(dev_data.sent)

        print("Dev Loss:{}\tmicro_f1:{}".format(avg_loss.item(), dev_micro_f1))
        print("micro_f1s:", end="")
        for i in range(take):
            precision = dev_metric["TP"][i].item() / (dev_metric["TP"][i].item()+dev_metric["FP"][i].item()+1e-10)
            recall = dev_metric["TP"][i].item() / (dev_metric["TP"][i].item()+dev_metric["FN"][i].item()+1e-10)
            print("{}".format(2*precision*recall / (precision+recall+1e-10)), end="\t")
        print("\n=====================================")

        if dev_micro_f1 > best_f1:
            best_f1 = dev_micro_f1

torch.save(model.state_dict(), "./model/model_{}_state".format(seed))