import os
import time
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from lib.utils import MoleculeDGL, MoleculeDatasetDGL, Dictionary, Molecule


def LapEig_positional_encoding(g, pos_enc_dim):
    n = g.number_of_nodes() # Number of nodes

    Adj = g.adj().to_dense() # Adjacency matrix
    Dn = ( g.in_degrees()** -0.5 ).diag() # Inverse and sqrt of degree matrix
    Lap = torch.eye(n) - Dn.matmul(Adj).matmul(Dn) # Laplacian operator
    EigVal, EigVec = torch.linalg.eig(Lap) # Compute full EVD
    EigVal, EigVec = EigVal.real, EigVec.real # make eig real
    EigVec = EigVec[:, EigVal.argsort()] # sort in increasing order of eigenvalues

    pe = torch.zeros(n, pos_enc_dim) # initialize positional encoding
    if pos_enc_dim >= n:
        EigVec = EigVec[:, 1:]
        pe[:, : n - 1].copy_(EigVec)
    else:
        pe.copy_(EigVec[:, 1: pos_enc_dim + 1])

    return pe


def main():
    # Select dataset
    dataset_name = 'QM9_1.4k'
    data_folder_pytorch = 'dataset/QM9_1.4k_pytorch/' 
    data_folder_dgl = 'dataset/QM9_1.4k_dgl/'

    # Load the number of atom and bond types
    with open(data_folder_pytorch + "atom_dict.pkl" ,"rb") as f: 
        num_atom_type = len(pickle.load(f))
    with open(data_folder_pytorch + "bond_dict.pkl" ,"rb") as f: 
        num_bond_type = len(pickle.load(f))
    
    print(num_atom_type)
    print(num_bond_type)

    # Load the DGL datasets
    datasets_dgl = MoleculeDatasetDGL(dataset_name, data_folder_dgl)
    
    def add_pos_enc(this_dataset):
        length = len(this_dataset)
        for i in tqdm(range(length)):
            g, _ = this_dataset[i]
            pe = LapEig_positional_encoding(g, 8)
            g.ndata['pos_enc'] = pe

    # Saving
    start = time.time()
    
    data_folder_dgl = 'dataset/QM9_1.4k_dgl_PE'
    print('Saving dgl molecules to folder : ' + data_folder_dgl )
    
    print("Processing test set...")
    add_pos_enc(datasets_dgl.test)
    with open(data_folder_dgl + "/test_dgl.pkl", "wb+") as f:
        pickle.dump(datasets_dgl.test,f)

    print("Processing val set...")
    add_pos_enc(datasets_dgl.val)
    with open(data_folder_dgl +  "/val_dgl.pkl", "wb+") as f:
        pickle.dump(datasets_dgl.val,f)
    
    print("Processing train set...")
    add_pos_enc(datasets_dgl.train)
    with open(data_folder_dgl +  "/train_dgl.pkl", "wb+") as f:
        pickle.dump(datasets_dgl.train,f)
    
    print('Time:',time.time()-start)
    print('Done!')
    

def generate_pe_stat():
    dataset_name = 'QM9_1.4k'
    data_folder_pytorch = 'dataset/QM9_1.4k_pytorch/' 
    data_folder_dgl = 'dataset/QM9_1.4k_dgl_PE/'

    # Load the number of atom and bond types
    with open(data_folder_pytorch + "atom_dict.pkl" ,"rb") as f: 
        num_atom_type = len(pickle.load(f))
    with open(data_folder_pytorch + "bond_dict.pkl" ,"rb") as f: 
        num_bond_type = len(pickle.load(f))
    
    print(num_atom_type)
    print(num_bond_type)

    # Load the DGL datasets
    datasets_dgl = MoleculeDatasetDGL(dataset_name, data_folder_dgl)
    trainset, valset, testset = datasets_dgl.train, datasets_dgl.val, datasets_dgl.test

    cat_dict = {}
    def add_cat_dict(this_dataset):
        length = len(this_dataset)
        for i in tqdm(range(length)):
            g, _ = this_dataset[i]
            pe = g.ndata['pos_enc']
            n = g.number_of_nodes()
            
            if n not in cat_dict:
                cat_dict[n] = []
            
            m = min(n - 1, 8)
            cat_dict[n].append(pe[:, :m])
    
    add_cat_dict(trainset)
    add_cat_dict(valset)
    add_cat_dict(testset)
    
    for k, v in cat_dict.items():
        print("Size: ", k, "Number of molecules: ", len(v))
        agg = torch.cat(v, dim=0).abs()
        mean = agg.mean(dim=0)
        std = agg.std(dim=0)
        print("Mean: ", mean)
        print("Std: ", std)
    
    print(cat_dict[4][0])



if __name__ == '__main__':
    # main()
    generate_pe_stat()