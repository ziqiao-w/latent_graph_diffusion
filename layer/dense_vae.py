import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layer.dense import DenseTransformerLayer, sym_tensor

class DenseVAE(nn.Module):
    def __init__(self, d_rep, d_model, n_heads, n_layers, n_atom_type, n_bond_type, n_max_pos, dropout=0.0):
        super().__init__()
        self.d_rep = d_rep
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_atom_type = n_atom_type
        self.n_bond_type = n_bond_type
        self.n_max_pos = n_max_pos
        self.dropout = dropout
        
        self.embedding_pos_enc = nn.Embedding(n_max_pos, d_model)
        self.embedding_h = nn.Embedding(n_atom_type, d_model)
        self.embedding_e = nn.Embedding(n_bond_type, d_model)
        
        self.encoder_layers = nn.ModuleList([ 
            DenseTransformerLayer(
                d_model=d_model,
                n_heads=d_model,
                dropout=dropout
            ) for _ in range(n_layers)])
        
        self.decoder_layers = nn.ModuleList([ 
            DenseTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)])
        
        self.proj_q_mu = nn.Linear(d_model, d_rep)
        self.proj_q_logvar = nn.Linear(d_model, d_rep)

        self.proj_d_model = nn.Linear(d_rep, d_model)

        self.ln_x = nn.LayerNorm(d_model)
        self.proj_x = nn.Linear(d_model, n_atom_type)

        self.ln_e = nn.LayerNorm(d_model)
        self.proj_e = nn.Linear(d_model, n_bond_type)

        self.drop_x_emb = nn.Dropout(dropout)
        self.drop_e_emb = nn.Dropout(dropout)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def encode(self, h, e, bs, n):
        
        h = self.embedding_h(h)  # [bs, n, d_model]
        h = self.drop_x_emb(h)

        pe_x = torch.arange(0, n, device=self.device).repeat(bs, 1)  # [bs, n]
        pe_x = self.embedding_pos_enc(pe_x)  # [bs, n, d_model]        

        e = self.embedding_e(e)  # [bs, n, n, d_model]
        e = sym_tensor(e)
        e = self.drop_e_emb(e)
        
        for layer in self.encoder_layers:
            h, e = layer(h, e)
        
        hz = h.mean(dim=1)  # [bs, d_model]

        q_mu = self.proj_q_mu(hz)           # [bs, d_rep]
        q_logvar = self.proj_q_logvar(hz)   # [bs, d_rep]
        q_std = torch.exp(0.5 * q_logvar)   # [bs, d_rep]
        eps = torch.randn_like(q_std)       # [bs, d_rep]
        z = q_mu + eps * q_std              # [bs, d_rep]
        
        return z, q_mu, q_logvar, pe_x
    
    def decode(self, z, bs, n, pe_x=None):
        if pe_x is None:
            pe_x = torch.arange(0, n, device=self.device).repeat(bs, 1)  # [bs, n]
            pe_x = self.embedding_pos_enc(pe_x)  # [bs, n, d_model]   

        z = self.proj_d_model(z)
        x = z.unsqueeze(1).expand(bs, n, self.d_model)
        x = x + pe_x

        e = z.unsqueeze(1).unsqueeze(2).expand(bs, n, n, self.d_model)
        e = sym_tensor(e)

        e = e + pe_x.unsqueeze(1) + pe_x.unsqueeze(2)
        e = sym_tensor(e)

        for layer in self.decoder_layers:
            x, e = layer(x, e)
        
        x = self.ln_x(x)
        x = self.proj_x(x)

        e = self.ln_e(e)
        e = self.proj_e(e)

        return x, e
        
    def forward(self, h, e, bs, n):
        if self.training:
            z, q_mu, q_logvar, pe = self.encode(h, e, bs, n)
        else:
            z = torch.randn(bs, self.d_rep, device=self.device)
            q_mu, q_logvar, pe = None, None, None
        
        x, y = self.decode(z, bs, n, pe)

        return x, y, q_mu, q_logvar
        
    def loss(self, x_pred, e_pred, q_mu, q_logvar, x_target, e_target, bs, n):
        loss_data = self.loss_fn(x_pred.view(bs * n, self.n_atom_type), x_target.view(bs * n))
        loss_data = loss_data + self.loss_fn(e_pred.view(bs * n * n, self.n_bond_type), e_target.view(bs * n * n))
        loss_KL = -0.15 * torch.mean( 1.0 + q_logvar - q_mu.pow(2.0) - q_logvar.exp() )
        loss_VAE = 1.0 * loss_data + loss_KL
        
        return loss_VAE
