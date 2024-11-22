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
import math
import sys; sys.path.insert(0, 'lib/')
from lib.molecules import Dictionary, Molecule, from_pymol_to_smile
# from vae import VAE
from data_prep import MoleculeSampler
import tqdm
from DiT import DiT
from EMA import EMA
# from layer.dense_vae import DenseVAE
from layer.predefined_vae import VAE
# PyTorch version and GPU
print(torch.__version__)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
  device= torch.device("cuda:0") # use GPU
else:
  device= torch.device("cpu")
print(device)

# batchsize
bs = 256
# latent dim
dz = 64
start = time.time()

data_folder_pytorch = 'dataset/QM9_pytorch/'

with open(data_folder_pytorch+"atom_dict.pkl","rb") as f:
    atom_dict=pickle.load(f)
with open(data_folder_pytorch+"bond_dict.pkl","rb") as f:
    bond_dict=pickle.load(f)
with open(data_folder_pytorch+"test_pytorch.pkl","rb") as f:
    test=pickle.load(f)
with open(data_folder_pytorch+"val_pytorch.pkl","rb") as f:
    val=pickle.load(f)
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

test_group  = group_molecules_per_size(test)
val_group   = group_molecules_per_size(val)
train_group = group_molecules_per_size(train)

# largest size of molecule in the trainset
max_mol_sz = max(list( train_group.keys()))

vae = VAE(max_mol_sz=max_mol_sz, num_atom_type=num_atom_type, num_bond_type=num_bond_type, device=device)
vae = vae.to(device)
vae.load_state_dict(torch.load('model_weight/my_vae_model.pth'))
vae.eval()

def display_num_param(net, model_name):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('Number of '+ model_name + ' parameters: {} ({:.2f} million)'.format(nb_param, nb_param/1e6))
    return nb_param/1e6
_ = display_num_param(vae, "VAE")

net = DiT(
            img_resolution=8,
            patch_size=8,
            in_channels=1,
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            label_dropout=0.,
            num_classes=max_mol_sz + 1,
            learn_sigma=False,
            out_channels=1, 
        )
net = net.to(device)

_ = display_num_param(net, "DiT")

drop = 0.0 # dropout value
num_mol_size = 20
num_warmup = 2 * max( num_mol_size, len(train) // bs ) # 4 epochs * max( num_mol_size=20, num_mol/batch_size)
print('num_warmup :',num_warmup)


init_lr = 0.0002
optimizer = torch.optim.AdamW(net.parameters(), lr=init_lr)

#EMA
# optimizer = EMA(optimizer, ema_decay=0.9999)
# scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: min((t+1)/num_warmup, 1.0) ) # warmup scheduler
# scheduler_tracker = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=1, verbose=True) # tracker scheduler
nb_epochs = 1000
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs, eta_min=1e-5)
num_warmup_batch = 0

for epoch in tqdm.tqdm(range(nb_epochs)):
    running_loss = 0.0
    num_batches = 0
    net.train()
    sampler = MoleculeSampler(train_group, bs)
    count = 0
    while(not sampler.is_empty()):

        sz = sampler.choose_molecule_size()
        indices = sampler.draw_batch_of_molecules(sz) 
        batch_x1 = torch.stack( [ train_group[sz][i].atom_type for i in indices] ).long().to(device) # [bs, n]
        batch_e1 = torch.stack( [ train_group[sz][i].bond_type for i in indices] ).long().to(device) # [bs, n, n]
        
        batch_z1 = vae.encode(batch_x1, batch_e1) # [bs, z_dim]
        t = torch.rand((batch_z1.size(0),), dtype=torch.float32, device=device, )
        t = t.view(-1, 1, 1, 1)
        batch_z1 = batch_z1.reshape(batch_z1.size(0), 1, 8, 8) # [bs, z_dim] -> [bs, 1, 8, 8]
        batch_z0 = torch.randn_like(batch_z1).to(device) # [bs, 1, dz]
        batch_zt = t * batch_z1 + (1-t) * batch_z0 # [bs, 1, dz]
        u = batch_z1 - batch_z0 # [bs, 1, dz]
        sz = sz * torch.ones((batch_z1.size(0),), dtype=torch.long, device=device)
        v = net(t.squeeze(), batch_zt, y=sz) # [bs, 1, dz]
        # loss = (u - v) ** 2
        # loss = loss.mean()
        loss = torch.nn.MSELoss()(u, v)

        # net.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # grad_norm_clip=1.0
        optimizer.step()

        # if num_warmup_batch < num_warmup:
        #     scheduler_warmup.step() # warmup scheduler
        # num_warmup_batch += 1
        scheduler.step()
        # Compute stats
        running_loss += loss.detach().item()
        num_batches += 1 
        
        
    # Average stats
    mean_loss = running_loss/ num_batches       
    # if num_warmup_batch >= num_warmup:
    #     scheduler_tracker.step(mean_loss) # tracker scheduler defined w.r.t. loss value
    elapsed = (time.time()-start)/60    
    if not epoch%50:
        print('epoch= %d \t time= %.4f min \t lr= %.7f \t loss= %.4f' % (epoch, elapsed, optimizer.param_groups[0]['lr'], mean_loss) )

    # Check lr value
    if optimizer.param_groups[0]['lr'] < 10**-6: 
      print("\n lr is equal to min lr -- training stopped\n") 
      break

# Save the model
# optimizer.swap_parameters_with_ema(store_params_in_ema=True)
torch.save(net.state_dict(), 'model_weight/flow_model_10k.pth')
print("Model saved")