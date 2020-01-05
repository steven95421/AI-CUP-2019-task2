import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
from transformers import AdamW, WarmupLinearSchedule

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import tqdm
import csv
import random
import matplotlib.pyplot as plt
import os
import sys

if not os.path.exists("model/task2/model_{}_state".format(sys.argv[1])):
    print("NOT FOUND")
    sys.exit()

import math

class gelu(nn.Module):
    
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * cdf

    
class MutiLabelModel(nn.Module):
    
    def __init__(self, encoder, emb_size=1024, out_size=4, ce_size=4, hidden=256): # hidden=256
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
            nn.Linear(self.fn_size//2, out_size),
        )
#         self.out_fn2 = nn.Sequential(
#             gelu(),
#             nn.Dropout(0.2),
#             nn.Linear(self.fn_size//8+ce_size, out_size)
#         )
        
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

use_cuda = torch.cuda.is_available()
if use_cuda:
    print("using cuda!")
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
                    self.category.append(row[4].split("/"))
                    self.category[-1] = [cat.split(".")[0] for cat in self.category[-1]]
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
        
        
data_dir = "data"
test_data = Data(idxs=[i for i in range(20000)], is_test=True)
test_data.readData(data_dir+"/task2_private_testset.csv")
    
# load pre-trained bert
print("loading bert...")
tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased')
    
MAX_LEN = 512
seed = int(sys.argv[1])
take = 4
print("seed:", seed)
device = torch.device("cuda:"+sys.argv[2])
print("device: {}".format(device))

encoder = BertModel.from_pretrained('scibert_scivocab_uncased')
model = MutiLabelModel(encoder, 768, take)
model.load_state_dict(torch.load("./model/task2/model_{}_state".format(seed), map_location=device))
model = model.to(device)
model = model.eval()

test_loss = 0.0
thrld = np.ones((1,4))*0.35
# thrld = np.ones((1,4))*0.5
# thrld[0][0] = 0.35
# thrld[0][1] = 0.3
# thrld[0][2] = 0.25
# thrld[0][3] = 0.35
thrld_ten = torch.from_numpy(thrld).float().to(device)

preds = []
ids = []

with torch.no_grad():

    for sidx, s in tqdm.tqdm(enumerate(test_data.sent), total=20000):

        # category
#         tmp = []
#         for cat in test_data.category[sidx]:
#             if cat in list(category_counts.keys()):
#                 tmp.append(cat)
#         test_data.category[sidx] = tmp.copy()
#         category_emb = [sum([category_counts[cat][i] for cat in test_data.category[sidx]]) for i in range(4)]
#         category_emb = [ce/sum(category_emb) for ce in category_emb]
#         category_emb = torch.tensor([category_emb]).float()
#         if use_cuda:
#             category_emb = category_emb.to(device)

        # abstract
        indexed_tokens = tokenizer.add_special_tokens_single_sequence(tokenizer.encode(s))[:MAX_LEN]
        input_ids = torch.tensor([indexed_tokens]).long()
        if use_cuda:
            input_ids = input_ids.to(device)

        segments_ids = [ 0 for i in range(len(indexed_tokens)) ]
        segments_ids = torch.tensor([segments_ids]).long()
        if use_cuda:
            segments_ids = segments_ids.to(device)
        
        out = model(input_ids, segments_ids) # , category_emb)

        out = torch.sigmoid(out)
        pred1 = (out > thrld_ten.expand(torch.Size([1, 4]))).float()
        max_idx = torch.argmax(out, 1, keepdim=True)
        one_hot = torch.FloatTensor(out.shape).to(device)
        one_hot.zero_()
        pred2=one_hot.scatter_(1, max_idx, 1)
        pred=(pred1+pred2>=1).float()
        
        preds.append(pred.cpu().tolist())
        ids.append("T"+str(sidx+20001))

preds = [pred[0] for pred in preds]

for i in range(len(preds)):
    for j in range(len(preds[i])):
        preds[i][j] = int(preds[i][j])
        
with open("results/task2/result_{}.csv".format(seed), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order_id", "THEORETICAL", "ENGINEERING", "EMPIRICAL", "OTHERS"])
    for i in range(1, 20001):
        w.writerow(["T"+"0"*(5-len(str(i)))+str(i), 0, 0, 0, 0])
    for i in range(len(ids)):
        w.writerow([ids[i]]+preds[i])