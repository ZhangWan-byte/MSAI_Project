import sys
import json
import pickle
import random
import numpy as np

import torch
import torch.nn as nn

# sys.path.append("../")
from utils import *

set_seed(42)

def split_data(data_path = "../SAbDab_mean/sabdab_all.json", save_path="../SAbDab_mean/"):

    # load data
    with open(data_path, "r") as fin:
        lines = fin.read().strip().split('\n')
        items = [json.loads(s) for s in lines]
    fin.close()

    print("all pdb: ", len(items))

    # filter with H/L/A    
    items = list(filter(lambda x: x["heavy_chain_seq"]!="" and x["light_chain_seq"]!="" and x["antigen_seqs"]!=[], items))
    print("pdb with heavy/light/antigen: ", len(items))

    # 8:1:1 split
    random.shuffle(items)
    train, val, test = items[:int(len(items)*0.8)], items[int(len(items)*0.8):int(len(items)*0.9)], items[int(len(items)*0.9):]
    print(len(train), len(val), len(test))

    # save
    with open(os.path.join(save_path, "train.json"), "w") as fout:
        for item in train:
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')
    fout.close()

    with open(os.path.join(save_path, "val.json"), "w") as fout:
        for item in val:
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')
    fout.close()

    with open(os.path.join(save_path, "test.json"), "w") as fout:
        for item in test:
            item_str = json.dumps(item)
            fout.write(f'{item_str}\n')
    fout.close()


if __name__ == "__main__":
    split_data(data_path="../SAbDab_mean/sabdab_all.json", save_path="../SAbDab_mean/")
