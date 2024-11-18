# Libraries
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

from lib.utils import Dictionary, Molecule, from_pymol_to_smile
from layer.predefined_vae import VAE
from tqdm import tqdm
from data_prep import sample_molecule_size

# PyTorch version and GPU
print(torch.__version__)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  device= torch.device("cuda:0") # use GPU
else:
  device= torch.device("cpu")
print(device)

data_folder_pytorch = 'dataset/QM9_1.4k_pytorch/'

with open(data_folder_pytorch+"atom_dict.pkl","rb") as f:
    atom_dict=pickle.load(f)
with open(data_folder_pytorch+"bond_dict.pkl","rb") as f:
    bond_dict=pickle.load(f)
with open(data_folder_pytorch+"train_pytorch.pkl","rb") as f:
    train=pickle.load(f)

num_atom_type = len(atom_dict.idx2word)
num_bond_type = len(bond_dict.idx2word)

def group_molecules_per_size(dataset):
    mydict={}
    for mol in dataset:
        if len(mol) not in mydict:
            mydict[len(mol)]=[]
        mydict[len(mol)].append(mol)
    return mydict

train_group = group_molecules_per_size(train)

# largest size of molecule in the trainset
max_mol_sz = max(list( train_group.keys()))

net = VAE(max_mol_sz=max_mol_sz, num_atom_type=num_atom_type, num_bond_type=num_bond_type, device=device)
net = net.to(device)
net.load_state_dict(torch.load('model_weight/vae_d64.pth'))
net.eval()

dz = 64 # number of dimensions for the compressed representation
# compute percentage of valid molecules
def is_reasonable_atom_count(mol, min_atoms=3, max_atoms=50):
    num_atoms = mol.GetNumAtoms()
    return min_atoms <= num_atoms <= max_atoms

def is_valid_generated_smiles(smiles, min_atoms=3, max_atoms=50):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except:
        return False
    return is_reasonable_atom_count(mol, min_atoms, max_atoms)

def compute_perc_valid_molecules(net, sampler_size, num_gen_mol=1000, num_generated_mols_per_batch=100):
    # num_atom = 9 # QM9
    num_batches = num_gen_mol // num_generated_mols_per_batch 
    num_valid_mol = 0
    list_valid_mol = []
    list_mol = []
    start = time.time()
    for idx in tqdm(range(num_batches)):
        net.eval()
        with torch.no_grad():  
            num_atom_sampled = sampler_size.choose_molecule_size() # sample the molecule size
            # num_atom_sampled = num_atom # same size
            batch_x_0, batch_e_0 = net.decode(torch.Tensor(num_generated_mols_per_batch, dz).normal_(mean=0, std=1), num_generated_mols_per_batch, num_atom_sampled) # [bs, n, num_atom_type], [bs, n, n, num_bond_type]
            # batch_x_0, batch_e_0, _, _  = net(0, 0, False, num_generated_mols_per_batch, num_atom_sampled) # [bs, n, num_atom_type], [bs, n, n, num_bond_type]
            batch_x_0 = torch.max(batch_x_0,dim=2)[1]  # [bs, n] 
            batch_e_0 = torch.max(batch_e_0,dim=3)[1]  # [bs, n, n]
            x_hat = batch_x_0.detach().to('cpu')
            e_hat = batch_e_0.detach().to('cpu')
            for x,e in zip(x_hat,e_hat):
                pymol = Molecule(num_atom_sampled, num_atom_type)
                pymol.atom_type = x
                pymol.bond_type = e
                smile = from_pymol_to_smile(pymol, atom_dict, bond_dict)
                list_mol.append(smile)
                # mol = Chem.MolFromSmiles(smile)
                if is_valid_generated_smiles(smile, min_atoms=num_atom_sampled, max_atoms=num_atom_sampled):
                    list_valid_mol.append(smile)
                    num_valid_mol += 1
    perc_valid_molecules = 100*num_valid_mol/num_gen_mol
    line = '\t num_gen_mol= ' + str(num_gen_mol) + '   time(min)= ' + str((time.time()-start)/60)[:5] + '   perc valid molecules= ' + str(perc_valid_molecules)[:6]
    return perc_valid_molecules, list_mol, line, list_valid_mol


sampler_size = sample_molecule_size(train_group)
perc_valid_mol, list_mol, line, list_valid_mol = compute_perc_valid_molecules(net, sampler_size)
print('percentage of valid molecules')
print(line)
print(list_valid_mol[:10])

num_valid_mol = len(list_valid_mol)
num_print_mol = 16
list_idx = torch.randperm(num_valid_mol)[:num_print_mol] 
print(list_idx)

# from rdkit.Chem import Draw
# list_valid_mol_img = [ Draw.MolToImage(Chem.MolFromSmiles(list_valid_mol[idx]),size=(512, 512)) for idx in list_idx ]
# 
# # Plot
# plt.figure(1, dpi=200)
# figure, axis = plt.subplots(4, 4)
# figure.set_size_inches(16,16)
# i,j,cpt=0,0,0; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=1,0,1; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=2,0,2; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=3,0,3; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=0,1+0,4; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=1,1+0,5; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=2,1+0,6; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=3,1+0,7; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=0,2+0,8; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=1,2+0,9; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=2,2+0,10; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=3,2+0,11; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=0,3+0,12; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=1,3+0,13; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=2,3+0,14; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# i,j,cpt=3,3+0,15; axis[i,j].imshow(list_valid_mol_img[cpt]); axis[i,j].set_title("Generated w/ VAE"); axis[i,j].axis('off')
# plt.savefig('generated_molecules.png')

print('num_generated_mol',len(list_mol))
num_unique_mol = 0
for idx,mol in enumerate(list_mol):
    list_tmp = list_mol.copy()
    list_tmp.pop(idx)
    if mol not in list_tmp:
        num_unique_mol += 1
print('num_unique_mol, num_mol:',num_unique_mol, len(list_mol))
perc_unique_mol = 100*num_unique_mol/len(list_mol)
print('perc unique molecules among the generated molecules:', str(perc_unique_mol)[:6])

list_train_mol = [] 
for idx in range(len(train)):
    list_train_mol.append(train[idx].smile) 
print('num_train_mol',len(list_train_mol))
print(list_train_mol[:10])
num_unique_mol = 0
for mol in list_mol:
    if mol not in list_train_mol:
        num_unique_mol += 1
print('num_unique_mol, num_mol:',num_unique_mol, len(list_mol))
perc_novel_mol = 100 * num_unique_mol / len(list_mol)
print('perc of novelty:', str(perc_novel_mol)[:6])

# check the number of unique novel molecules in the generated molecules
valid_mol_set = set(list_valid_mol)
num_unique_noval_mol = 0
train_mol_set = set(list_train_mol)
for mol in valid_mol_set:
    if mol not in train_mol_set:
        num_unique_noval_mol += 1

print('num_unique_noval_mol: ',num_unique_noval_mol)
print(f'final score: {100 * num_unique_noval_mol / len(list_mol): .2f}')
