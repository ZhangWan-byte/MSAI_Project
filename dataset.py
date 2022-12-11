import os
import copy
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# codes borrowed from
# https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(seed=42)


def get_mask(seq, substrs):
    # seq: [Hseq]
    # substrs: [H1, H2, H3]
    # return [mask_H1, mask_H2, mask_H3]
    mask = [0] * len(seq)
    m = 1
    span = []
    for substr in substrs:
        if len(span)==0:
            start = seq.find(substr)
            end = start+len(substr)
        else:
            start = seq.find(substr, span[-1][-1], -1)
            end = start+len(substr)
        span.append((start, end))
        for idx in range(len(mask)):
            if idx>=start and idx<end:
                mask[idx] = m
        m += 1
    
    return mask, span


def process(data):
    # input: data
    # :data: dict containing original processed data

    # return: [train, val, test]
    # :train: {"X":X, "S":S, "mask":mask}
    # :X: [seq_len, 4, 3], coordinates of N, CA, C, O. Missing data are set to 0
    # :S: [seq_len], indices of each residue
    # :mask: [Hmask, Lmask] - string of cdr labels, 0 for non-cdr residues, 1 for cdr1, 2 for cdr2, 3 for cdr3 

    data = copy.deepcopy(data)

    for i in range(len(data)):
        Hseq = data[i]["Hseq"][0]
        Lseq = data[i]["Lseq"][0]
        Aseq = "/".join(data[i]["Aseq"])    # SEP "/"

        H1, H2, H3 = data[i]["H1"], data[i]["H2"], data[i]["H3"]
        L1, L2, L3 = data[i]["L1"], data[i]["L2"], data[i]["L3"]

        Hpos = data[i]["Hpos"]
        Lpos = data[i]["Lpos"]
        Apos = [achain for achain in data[i]["Apos"]]
        # insert zeros as positions of "SEP" -- "/" for antigen chain
        if len(Apos)>1:
            Apos = [np.zeros((4,3)).astype(np.float32) if idx%2==0 else ele for idx,ele in enumerate(Apos)]
            Apos = Apos[:-1] if len(Apos)%2==0 else Apos

        Hmask, Hspan = get_mask(seq=Hseq, substrs=[data[i]["H1"], data[i]["H2"], data[i]["H3"]])
        Lmask, Lspan = get_mask(seq=Lseq, substrs=[data[i]["L1"], data[i]["L2"], data[i]["L3"]])

        # sanitisation through checking non-zero elements
        # for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        if np.count_nonzero(Hmask) != len(data[i]["H1"])+len(data[i]["H2"])+len(data[i]["H3"]):
            print("Hcdr disalignment: {} at position {}".format(data[i]["pdb"], i))

        if np.count_nonzero(Lmask) != len(data[i]["L1"])+len(data[i]["L2"])+len(data[i]["L3"]):
            print("Lcdr disalignment: {} at position {}".format(data[i]["pdb"], i))

        Hpos_cdr = []
        for idx in range(3):
            Hpos_cdr.append(Hpos[Hspan[idx][0]:Hspan[idx][1]])
            if idx < 2:
                Hpos_cdr.append(np.zeros((4,3)))
        
        Lpos_cdr = []
        for idx in range(3):
            Lpos_cdr.append(Lpos[Lspan[idx][0]:Lspan[idx][1]])
            if idx < 2:
                Lpos_cdr.append(np.zeros((4,3)))
        
        pos = Hpos_cdr + [np.zeros((4,3))] + Lpos_cdr
        seq = H1 + "/" + H2 + "/" + H3 + "/" + L1 + "/" + L2 + "/" + L3
        

        data[i] = {"X":pos, "S":seq, "mask":[Hmask, Lmask]}

    random.seed(42)
    random.shuffle(data)

    # train:val:test = 7:1:2
    train, test = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
    train, val = train[:int(0.7*len(data))], train[int(0.7*len(data)):]

    return train, val, test


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_length=128, is_train=True, is_kfold=False, kfold=10, val_fold=0):
        self.data = data
        self.total_length = len(data)

        if is_kfold==True:
            self.kfold = kfold
            self.val_fold = val_fold
        
            self.data_folds = []
            self.label_folds = []
            for k in range(kfold):
                data_tmp = self.data[k*int(0.1*self.data.shape[0]):(k+1)*int(0.1*self.data.shape[0])]
                label_tmp = self.label[k*int(0.1*self.label.shape[0]):(k+1)*int(0.1*self.label.shape[0])]
                self.data_folds.append(data_tmp)
                self.label_folds.append(label_tmp)
        
            self.test_data = self.data_folds.pop(holdout_fold)
            self.test_label = self.label_folds.pop(holdout_fold)
            self.train_data = pd.concat(self.data_folds)
            self.train_label = torch.hstack(self.label_folds)
        else:
            pass


    def __len__(self):
        if self.is_train==True:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
    
    def __getitem__(self, idx):
        if self.is_train==True:
            return self.train_data.iloc[idx][0], self.train_data.iloc[idx][1], self.train_label[idx]
        else:
            return self.test_data.iloc[idx][0], self.test_data.iloc[idx][1], self.test_label[idx]


if __name__ == "__main__":
    print("hello world!")

    