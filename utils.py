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


AA_abbr_alias = {
    "MSE": "M",    # abbr of MET
    "FTR": "W",    # a type of TRP
    "OAS": "S",    # a type of SER
    "TYS": "Y",    # a type of TYR
    "TPO": "T",    # a typr of THR
    'M': 'MSE', 
    'W': 'FTR', 
    'S': 'OAS', 
    'Y': 'TYS', 
    'T': 'TPO'
}

def to_abbr(aa):
    if aa in AA_abbr:
        return AA_abbr[aa]
    elif aa in AA_abbr_alias:
        return AA_abbr_alias[aa]
    else:
        return AA_abbr["UNK"]


def get_residue_seqs(pdb_path="../../../MSAI_Project/SAbDab_20221124/all_structures/imgt/1mhp.pdb", chains=["H"], all_chains=None):
    
    # with open(pdb_path) as f:
    #     lines = f.readlines()
    # f.close()

    p = PDBParser()

    structure = p.get_structure('input', pdb_path)

    chains_seqs = {}
    
    for chain in chains:

        if chain not in all_chains:
            chain = chain.lower() if chain.lower() in all_chains else chain.upper()
                
        chains_seqs[chain] = []
        AA_coord = []
        for residue in structure[0][chain]:
            # get abbr
            res_name = residue.get_resname()
            res_abbr = to_abbr(res_name)

            # get position
            # # use coord of previous res
            # if (residue.get_resname() not in AA_abbr) and (residue.get_resname() not in AA_abbr_alias):
            #     if len(AA_coord)>=1:
            #         AA_coord.append(AA_coord[-1])
            #     else:
            #         AA_coord.append(np.zeros((4, 3)))
            #         continue

            # store first atom in residue
            for temp_atom in residue:
                break

            # use coord of the first atom if missing
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

            pos = np.vstack([N_coord, CA_coord, C_coord, O_coord])

            # residue dict
            res = {"name":res_name, "abbr":res_abbr, "pos":pos}

            AA_coord.append(pos)

            chains_seqs[chain].append(res)

    return chains_seqs


def get_residue_pos(pdb_path="../SAbDab_20221124/all_structures/imgt/7k5y.pdb", chain="M"):
    p = PDBParser()

    structure = p.get_structure('input', pdb_path)

    AA_coord = []

    for residue in structure[0][chain]:
        # if residue.get_resname() not in AA_abbr:
        #     continue
    
        if (residue.get_resname() not in AA_abbr) and (residue.get_resname() not in AA_abbr_alias):
            if len(AA_coord)>=1:
                AA_coord.append(AA_coord[-1])
            else:
                AA_coord.append(np.zeros((4, 3)))
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