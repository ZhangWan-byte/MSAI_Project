import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="./data.json", seq_length=128, kfold=10, holdout_fold=0, is_train=True):
        self.is_train = is_train
        self.data = pickle.load(open(data_path, "rb"))
        self.total_length = len(data)

        self.processed_data = []
        self.processed_data = self.process()        

        self.data_folds = []
        self.label_folds = []
        for k in range(kfold):
            data_tmp = self.data[k*int(0.1*self.data.shape[0]):(k+1)*int(0.1*self.data.shape[0])]
            label_tmp = self.label[k*int(0.1*self.label.shape[0]):(k+1)*int(0.1*self.label.shape[0])]
            self.data_folds.append(data_tmp)
            self.label_folds.append(label_tmp)
            
#         print(self.data_folds[0])

        self.test_data = self.data_folds.pop(holdout_fold)
        self.test_label = self.label_folds.pop(holdout_fold)
        self.train_data = pd.concat(self.data_folds)
        self.train_label = torch.hstack(self.label_folds)


    # @staticmethod
    # def get_elements_from_mask(ele_list, mask):
    #     elements = "".join([data[0]["Hseq"][0][idx] if mask[idx]==1 else "" for idx in range(len(mask))])==data[0]["H1"], data[0]["H1"]
    #     return elements


    @staticmethod
    def get_mask(seq, substrs, is_Hchain=True):
        # seq: [Hseq]
        # substrs: [H1, H2, H3]
        # return [mask_H1, mask_H2, mask_H3]
        mask = [0] * len(seq)
        m = 1
        span = []
        for substr in substrs:
            start = seq.find(substr)
            end = start+len(substr)
            span.append((start, end))
            for idx in range(len(mask)):
                if idx>=start and idx<end:
                    mask[idx] = m
            m += 1
        
        return mask, span



    def process(self):
        # X: [seq_len, 4, 3], coordinates of N, CA, C, O. Missing data are set to 0
        # S: [seq_len], indices of each residue
        # Hmask/Lmask: string of cdr labels, 0 for non-cdr residues, 1 for cdr1, 2 for cdr2, 3 for cdr3 

        for i in range(len(self.data)):
            Hseq = self.data[i]["Hseq"][0]
            Lseq = self.data[i]["Lseq"][0]
            Aseq = "/".join(self.data[i]["Aseq"])    # SEP "/"

            H1, H2, H3 = self.data[i]["H1"], self.data[i]["H2"], self.data[i]["H3"]
            L1, L2, L3 = self.data[i]["L1"], self.data[i]["L2"], self.data[i]["L3"]

            Hpos = self.data[i]["Hpos"]
            Lpos = self.data[i]["Lpos"]
            Apos = [achain for achain in self.data[i]["Apos"]]
            # insert zeros as positions of "SEP" -- "/" for antigen chain
            if len(Apos)>1:
                Apos = [np.zeros((4,3)).astype(np.float32) if idx%2==0 else ele for idx,ele in enumerate(Apos)]
                Apos = Apos[:-1] if len(Apos)%2==0 else Apos

            Hmask, Hspan = self.get_mask(seq=Hseq, substrs=[data[i]["H1"], data[i]["H2"], data[i]["H3"]])
            Lmask, Lspan = self.get_mask(seq=Lseq, substrs=[data[i]["L1"], data[i]["L2"], data[i]["L3"]])

            # sanitisation through checking non-zero elements
            for cdr in ["H1", "H2", "H3", "L1", "L2", "L3"]:
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
            

            data[i] = {"X":pos, "S":seq}
            

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

    