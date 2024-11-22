# Libraries
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lib.utils import Dictionary, Molecule, from_pymol_to_smile
from layer.predefined_vae import VAE
from data_prep import MoleculeSampler
import tqdm
# from EMA import EMA

# Enable tf32 for faster FP32 training
torch.backends.cuda.matmul.allow_tf32 = True

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
def print_distribution(data):
    for nb_atom in range(9+1):
        try: 
            print('number of molecule of size {}: \t {}'.format(nb_atom, len(data[nb_atom])))
        except:
            pass
print('Train'); print_distribution(train_group)
# largest size of molecule in the trainset
max_mol_sz = max(list( train_group.keys()))

net = VAE(max_mol_sz=max_mol_sz, num_atom_type=num_atom_type, num_bond_type=num_bond_type, device=device)
net = net.to(device)

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('Number of parameters: {} ({:.2f} million)'.format(nb_param, nb_param/1e6))
    return nb_param/1e6
_ = display_num_param(net)

# Warmup 
num_mol_size = 20
num_warmup = 2 * max( num_mol_size, len(train) // bs ) # 4 epochs * max( num_mol_size=20, num_mol/batch_size)
print('num_warmup :',num_warmup)

init_lr = 0.0003
optimizer = torch.optim.Adam(net.parameters(), lr=init_lr)
#EMA
# optimizer = EMA(optimizer, ema_decay=0.9999)

scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda t: min((t+1)/num_warmup, 1.0) ) # warmup scheduler
scheduler_tracker = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True) # tracker scheduler

# Number of mini-batches per epoch
nb_epochs = 1000
num_warmup_batch = 0
print('num_warmup, nb_epochs :',num_warmup, nb_epochs)
# Run training epochs
start = time.time()
for epoch in tqdm.tqdm(range(nb_epochs)):

    running_loss = 0.0
    num_batches = 0
    
    net.train()

    # bs = 50
    sampler = MoleculeSampler(train_group, bs)
    while(not sampler.is_empty()):

        num_batches_remaining = sampler.compute_num_batches_remaining()
        sz = sampler.choose_molecule_size()
        indices = sampler.draw_batch_of_molecules(sz) 
        batch_x0 = torch.stack( [ train_group[sz][i].atom_type for i in indices] ).long().to(device) # [bs, n]
        batch_e0 = torch.stack( [ train_group[sz][i].bond_type for i in indices] ).long().to(device) # [bs, n, n]
        batch_x_pred, batch_e_pred, q_mu, q_logvar = net(batch_x0, batch_e0) # [bs, n], [bs, n, n]
        bs2 = batch_x_pred.size(0)
        loss_data = torch.nn.CrossEntropyLoss()(batch_x_pred.view(bs2*sz,num_atom_type), batch_x0.view(bs2*sz))
        loss_data = loss_data + torch.nn.CrossEntropyLoss()(batch_e_pred.view(bs2*sz*sz,num_bond_type), batch_e0.view(bs2*sz*sz))
        loss_KL = -0.15* torch.mean( 1.0 + q_logvar - q_mu.pow(2.0) - q_logvar.exp() )
        loss_VAE = 1.0 * loss_data + loss_KL
        loss = loss_VAE

        # net.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25) # grad_norm_clip=1.0
        optimizer.step()

        # Warmup scheduler for transformer layers
        if num_warmup_batch < num_warmup:
            scheduler_warmup.step() 
        num_warmup_batch += 1

        # Compute stats
        running_loss += loss.detach().item()
        num_batches += 1

    # Average stats
    mean_loss = running_loss/num_batches
    if num_warmup_batch >= num_warmup:
        scheduler_tracker.step(mean_loss) # tracker scheduler defined w.r.t. loss value
        num_warmup_batch += 1
    
    elapsed = (time.time()-start)/60
    if not epoch%5:
        print('epoch= %d \t time= %.4f min \t lr= %.7f \t loss= %.4f' % (epoch, elapsed, optimizer.param_groups[0]['lr'],mean_loss) )

    # Check lr value  
    if optimizer.param_groups[0]['lr'] < 10**-6:
      print("\n lr is equal to min lr -- training stopped\n")
      break

# Save the model
# optimizer.swap_parameters_with_ema(store_params_in_ema=True)
torch.save(net.state_dict(), 'model_weight/my_vae_model.pth')
print('Model saved')