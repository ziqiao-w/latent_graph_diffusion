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

# Global constants
dz = 64 # number of dimensions for the compressed representation
num_heads = 16 # number of heads in the transformer layer
d = 16 * num_heads # number of hidden dimensions
num_layers_encoder = 4 # number of transformer encoder layers
num_layers_decoder = 4 # number of transformer decoder layers
drop = 0.0 # dropout value


# Define VAE architecture with Transformers
class head_attention(nn.Module):
    def __init__(self, d, d_head):
        super().__init__()
        self.Q = nn.Linear(d, d_head)
        self.K = nn.Linear(d, d_head)
        self.E = nn.Linear(d, d_head)
        self.V = nn.Linear(d, d_head)
        self.sqrt_d = torch.sqrt(torch.tensor(d_head))
        self.drop_att = nn.Dropout(drop)
        self.Ni = nn.Linear(d, d_head)
        self.Nj = nn.Linear(d, d_head)
    def forward(self, x, e):
        ###############################################
        # YOUR CODE STARTS
        ###############################################
        Q = self.Q(x) # [bs, n, d_head]
        K = self.K(x) # [bs, n, d_head]
        V = self.V(x) # [bs, n, d_head]
        Q = Q.unsqueeze(2) # [bs, n, 1, d_head]
        K = K.unsqueeze(1) # [bs, 1, n, d_head]
        E = self.E(e) # [bs, n, n, d_head]
        Ni = self.Ni(x).unsqueeze(2) # [bs, n, 1, d_head]
        Nj = self.Nj(x).unsqueeze(1) # [bs, 1, n, d_head]
        e = Ni + Nj + E              # [bs, n, n, d_head]
        Att = (Q * e * K).sum(dim=3) / self.sqrt_d # [bs, n, n]
        Att = torch.softmax(Att, dim=1)            # [bs, n, n]
        Att = self.drop_att(Att)                   # [bs, n, n]
        x = Att @ V                  # [bs, n, d_head]
        ###############################################
        # YOUR CODE ENDS
        ###############################################
        return x, e                  # [bs, n, d_head], [bs, n, n, d_head]

class MHA(nn.Module):
    def __init__(self, d, num_heads):  
        super().__init__()
        d_head = d // num_heads
        self.heads = nn.ModuleList( [head_attention(d, d_head) for _ in range(num_heads)] )
        self.WOx = nn.Linear(d, d)
        self.WOe = nn.Linear(d, d)
        self.drop_x = nn.Dropout(drop)
        self.drop_e = nn.Dropout(drop)
    def forward(self, x, e):
        ###############################################
        # YOUR CODE STARTS
        ###############################################
        x_MHA = []
        e_MHA = []    
        for head in self.heads:
            x_HA, e_HA = head(x,e)            # [bs, n, d_head], [bs, n, n, d_head]
            x_MHA.append(x_HA)
            e_MHA.append(e_HA)
        x = self.WOx(torch.cat(x_MHA, dim=2)) # [bs, n, d]
        x = self.drop_x(x)                    # [bs, n, d]
        e = self.WOe(torch.cat(e_MHA, dim=3)) # [bs, n, n, d]
        e = self.drop_e(e)                    # [bs, n, n, d]
        ###############################################
        # YOUR CODE ENDS
        ###############################################
        return x, e                           # [bs, n, d], [bs, n, n, d]

class BlockGT(nn.Module):
    def __init__(self, d, num_heads):  
        super().__init__()
        self.LNx = nn.LayerNorm(d)
        self.LNe = nn.LayerNorm(d)
        self.LNx2 = nn.LayerNorm(d)
        self.LNe2 = nn.LayerNorm(d)
        self.MHA = MHA(d, num_heads)
        self.MLPx = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.MLPe = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.drop_x_mlp = nn.Dropout(drop)
        self.drop_e_mlp = nn.Dropout(drop)
    def forward(self, x, e):
        ###############################################
        # YOUR CODE STARTS
        ###############################################
        x = self.LNx(x)                 # [bs, n, d]
        e = self.LNe(e)                 # [bs, n, n, d]
        x_MHA, e_MHA = self.MHA(x, e)   # [bs, n, d], [bs, n, n, d]
        x = x + x_MHA                   # [bs, n, d]
        x = x + self.MLPx(self.LNx2(x)) # [bs, n, d]
        x = self.drop_x_mlp(x)          # [bs, n, d]
        e = e + e_MHA                   # [bs, n, n, d]
        e = e + self.MLPe(self.LNe2(e)) # [bs, n, n, d]
        e = self.drop_e_mlp(e)          # [bs, n, n, d]
        ###############################################
        # YOUR CODE ENDS
        ###############################################
        return x, e                     # [bs, n, d], [bs, n, n, d]

def sym_tensor(x):
    x = x.permute(0,3,1,2) # [bs, d, n, n]
    triu = torch.triu(x,diagonal=1).transpose(3,2) # [bs, d, n, n]
    mask = (triu.abs()>0).float()                  # [bs, d, n, n]
    x =  x * (1 - mask ) + mask * triu             # [bs, d, n, n]
    x = x.permute(0,2,3,1) # [bs, n, n, d]
    return x               # [bs, n, n, d]



class VAE(nn.Module): 
    def __init__(self, max_mol_sz, num_atom_type, num_bond_type, device= torch.device("cpu")):
        super().__init__()
        self.device = device
        self.pe_x = nn.Embedding(max_mol_sz, d)
        self.atom_emb = nn.Embedding(num_atom_type, d)
        self.bond_emb = nn.Embedding(num_bond_type, d)
        self.gt_enc_layers = nn.ModuleList( [BlockGT(d, num_heads) for _ in range(num_layers_encoder)] )
        self.gt_dec_layers = nn.ModuleList( [BlockGT(d, num_heads) for _ in range(num_layers_decoder)] )
        self.linear_q_mu     = nn.Linear(d, dz)
        self.linear_q_logvar = nn.Linear(d, dz)
        self.linear_p = nn.Linear(dz, d)
        self.ln_x_final = nn.LayerNorm(d)  
        self.linear_x_final = nn.Linear(d, num_atom_type)
        self.ln_e_final = nn.LayerNorm(d)  
        self.linear_e_final = nn.Linear(d, num_bond_type)
        self.drop_x_emb = nn.Dropout(drop)
        self.drop_e_emb = nn.Dropout(drop)
        self.drop_p_emb = nn.Dropout(drop)
    
    def encode(self, x, e):
        # with torch.no_grad():
            # input layer
            x = self.atom_emb(x)                   # [bs, n, d]
            bs2 = x.size(0); n = x.size(1)
            pe_x = torch.arange(0,n).to(self.device).repeat(bs2,1) # [bs, n] 
            pe_x = self.pe_x(pe_x)                            # [bs, n, d] 
            e = self.bond_emb(e)                   # [bs, n, n, d]
            e = sym_tensor(e)                      # [bs, n, n, d]
            x = self.drop_x_emb(x)                 # [bs, n, d]
            e = self.drop_e_emb(e)                 # [bs, n, n, d]        
            # GT layers   
            ###############################################
            # Get the distribution, including mean and logvar vectors
            # Sample `z` from the distribution via re-parameterization
            # YOUR CODE STARTS
            ###############################################
            for gt_enc_layer in self.gt_enc_layers:
                x, e = gt_enc_layer(x, e)          # [bs, n, d],  [bs, n, n, d]
                e = sym_tensor(e)                  # [bs, n, n, d]     
            # molecule token
            mol_token = x.mean(1)                  # [bs, d]
            # VAE Gaussian
            q_mu = self.linear_q_mu(mol_token)         # [bs, dz]
            q_logvar = self.linear_q_logvar(mol_token) # [bs, dz]
            q_std = torch.exp(0.5*q_logvar)            # [bs, dz]
            eps = torch.randn_like(q_std)              # [bs, dz]
            z = q_mu + eps * q_std                     # [bs, dz]

            return z
    
    def decode(self, z, num_gen=1, num_atom=9):
        # with torch.no_grad():
            z = z.to(self.device)
            bs_dec = num_gen; n = num_atom
            pe_x = torch.arange(0,n).to(self.device).repeat(bs_dec,1) # [bs, n] 
            pe_x = self.pe_x(pe_x)
            # decoder
            # input layer
            z = self.linear_p(z)             # [bs, d]  
            x = z.unsqueeze(1).repeat(1,n,1) # [bs, 1, d] => [bs, n, d]
            x = x + pe_x                     # [bs, n, d]
            e = z.unsqueeze(1).unsqueeze(2).repeat(1,n,n,1) # [bs, 1, 1, d] => [bs, n, n, d] 
            e = sym_tensor(e)                # [bs, n, n, d] 
            e = e + pe_x.unsqueeze(1) + pe_x.unsqueeze(2)       # [bs, n, n, d] 
            e = sym_tensor(e)                # [bs, n, n, d] 
            ###############################################
            # Code decoding process
            # YOUR CODE STARTS
            ###############################################
            # GT layers
            for gt_dec_layer in self.gt_dec_layers:
                x, e = gt_dec_layer(x, e)    # [bs, n, d],  [bs, n, n, d]
                e = sym_tensor(e)            # [bs, n, n, d] 
            ###############################################
            # YOUR CODE ENDS
            ###############################################
            # output
            x = self.ln_x_final(x)     # [bs, n, d]
            x = self.linear_x_final(x) # [bs, n, num_atom_type]
            e = self.ln_e_final(e)     # [bs, n, n, d] 
            e = self.linear_e_final(e) # [bs, n, n, num_bond_type]
            return x, e                    

    def forward(self, x, e, train=True, num_gen=1, num_atom=9):
        
        # encoder
        if train: # training phase
            # input layer
            x = self.atom_emb(x)                   # [bs, n, d]
            bs2 = x.size(0); n = x.size(1)
            pe_x = torch.arange(0,n).to(self.device).repeat(bs2,1) # [bs, n] 
            pe_x = self.pe_x(pe_x)                            # [bs, n, d] 
            e = self.bond_emb(e)                   # [bs, n, n, d]
            e = sym_tensor(e)                      # [bs, n, n, d]
            x = self.drop_x_emb(x)                 # [bs, n, d]
            e = self.drop_e_emb(e)                 # [bs, n, n, d]        
            # GT layers   
            ###############################################
            # Get the distribution, including mean and logvar vectors
            # Sample `z` from the distribution via re-parameterization
            # YOUR CODE STARTS
            ###############################################
            for gt_enc_layer in self.gt_enc_layers:
                x, e = gt_enc_layer(x, e)          # [bs, n, d],  [bs, n, n, d]
                e = sym_tensor(e)                  # [bs, n, n, d]     
            # molecule token
            mol_token = x.mean(1)                  # [bs, d]
            # VAE Gaussian
            q_mu = self.linear_q_mu(mol_token)         # [bs, dz]
            q_logvar = self.linear_q_logvar(mol_token) # [bs, dz]
            q_std = torch.exp(0.5*q_logvar)            # [bs, dz]
            eps = torch.randn_like(q_std)              # [bs, dz]
            z = q_mu + eps * q_std                     # [bs, dz]
            ###############################################
            # YOUR CODE ENDS
            ###############################################
            # bs_dec = bs; n = x.size(1) 
        else: # generation phase
            bs_dec = num_gen; n = num_atom
            pe_x = torch.arange(0,n).to(self.device).repeat(bs_dec,1) # [bs, n] 
            pe_x = self.pe_x(pe_x)                               # [bs, n, d] 
            z = torch.Tensor(bs_dec, dz).normal_(mean=0.0, std=1.0).to(self.device) # [b, dz]
            q_mu, q_logvar = None, None
            
        # decoder
        # input layer
        z = self.linear_p(z)             # [bs, d]  
        x = z.unsqueeze(1).repeat(1,n,1) # [bs, 1, d] => [bs, n, d]
        x = x + pe_x                     # [bs, n, d]
        e = z.unsqueeze(1).unsqueeze(2).repeat(1,n,n,1) # [bs, 1, 1, d] => [bs, n, n, d] 
        e = sym_tensor(e)                # [bs, n, n, d] 
        e = e + pe_x.unsqueeze(1) + pe_x.unsqueeze(2)       # [bs, n, n, d] 
        e = sym_tensor(e)                # [bs, n, n, d] 
        ###############################################
        # Code decoding process
        # YOUR CODE STARTS
        ###############################################
        # GT layers
        for gt_dec_layer in self.gt_dec_layers:
            x, e = gt_dec_layer(x, e)    # [bs, n, d],  [bs, n, n, d]
            e = sym_tensor(e)            # [bs, n, n, d] 
        ###############################################
        # YOUR CODE ENDS
        ###############################################
        # output
        x = self.ln_x_final(x)     # [bs, n, d]
        x = self.linear_x_final(x) # [bs, n, num_atom_type]
        e = self.ln_e_final(e)     # [bs, n, n, d] 
        e = self.linear_e_final(e) # [bs, n, n, num_bond_type] 

        return x, e, q_mu, q_logvar
