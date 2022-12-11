import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom


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


AA_abbr = {
     'A': 'ALA',
     'R': 'ARG',
     'N': 'ASN',
     'D': 'ASP',
     'C': 'CYS',
     'Q': 'GLN',
     'E': 'GLU',
     'G': 'GLY',
     'H': 'HIS',
     'I': 'ILE',
     'L': 'LEU',
     'K': 'LYS',
     'M': 'MET',
     'F': 'PHE',
     'P': 'PRO',
     'U': 'SEC',
     'S': 'SER',
     'T': 'THR',
     'W': 'TRP',
     'Y': 'TYR',
     'V': 'VAL',
     'ALA': 'A',
     'ARG': 'R',
     'ASN': 'N',
     'ASP': 'D',
     'CYS': 'C',
     'GLN': 'Q',
     'GLU': 'E',
     'GLY': 'G',
     'HIS': 'H',
     'ILE': 'I',
     'LEU': 'L',
     'LYS': 'K',
     'MET': 'M',
     'PHE': 'F',
     'PRO': 'P',
     'SEC': 'U',
     'SER': 'S',
     'THR': 'T',
     'TRP': 'W',
     'TYR': 'Y',
     'VAL': 'V',
     'UNK': '*',
     '*': 'UNK',
     'PAD': '#',
     '#': 'PAD',
     'SEP': '/',
     '/': 'SEP'
}


def to_abbr(aa):
    if aa in AA_abbr:
        return AA_abbr[aa]
    else:
        return AA_abbr["UNK"]


def get_residue_seqs(pdb_path="../../../MSAI_Project/SAbDab_20221124/all_structures/raw/1mhp.pdb", chains=["H", "L", "A"], all_chains=None):
    
    with open(pdb_path) as f:
        lines = f.readlines()
    f.close()

    chains_seqs = {}
    
    for chain in chains:

        if chain not in all_chains:
            chain = chain.lower() if chain.lower() in all_chains else chain.upper()
                
        chains_seqs[chain] = []
    
        for i in range(len(lines)):
            if lines[i][:6]=="SEQRES" and lines[i][11]==chain:
                chains_seqs[chain] += lines[i].split()[4:]
#     print(chains_seqs)

    results = []
    for k in chains_seqs.keys():
        results.append("".join(list(map(lambda x:to_abbr(x), chains_seqs[k]))))
    
    return results


def get_residue_pos(pdb_path="../SAbDab_20221124/all_structures/raw/7k5y.pdb", chain="M"):
    p = PDBParser()

    structure = p.get_structure('input', pdb_path)

    AA_coord = []
    chain = chain.lower() if chain not in [c.get_id() for c in structure[0].get_list()] else chain.upper()
    for residue in structure[0][chain]:
        if residue.get_resname() not in AA_abbr:
            continue
    
        for temp_atom in residue:
            break

        try:
            N_coord = residue['N'].get_coord()
        except:
            N_coord = temp_atom.get_coord()
        try:
            CA_coord = residue['CA'].get_coord()
        except:
            CA_coord = temp_atom.get_coord()
        try:
            C_coord = residue['C'].get_coord()
        except:
            C_coord = temp_atom.get_coord()
        try:
            O_coord = residue['O'].get_coord()
        except:
            O_coord = temp_atom.get_coord()

        coord = np.vstack([N_coord, CA_coord, C_coord, O_coord])
        AA_coord.append(coord)

    return AA_coord