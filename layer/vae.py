import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layer.transformer import GraphTransformerLayer
from layer.dense import DenseTransformerLayer, sym_tensor
from layer.common import sample_unit_sphere

class Transformer_VAE(nn.Module):
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
        
        self.embedding_pos_enc = nn.Linear(n_max_pos, d_model)
        self.embedding_h = nn.Embedding(n_atom_type, d_model)
        self.embedding_e = nn.Embedding(n_bond_type, d_model)
        
        self.in_h_dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([ 
            GraphTransformerLayer(
                in_dim=d_model,
                out_dim=d_model,
                num_heads=n_heads,
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
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def encode(self, g, h, e, pos_enc, bs, n):
        h = self.embedding_h(h)
        h = self.in_h_dropout(h)
        
        pos_enc = self.embedding_pos_enc(pos_enc)
        h = h + pos_enc

        e = self.embedding_e(e)
        
        for layer in self.encoder_layers:
            h, e = layer(g, h, e)
        
        h = h.reshape(bs, n, self.d_model)
        z = h.mean(dim=1)

        q_mu = self.proj_q_mu(z)
        q_logvar = self.proj_q_logvar(z)
        q_std = torch.exp(0.5 * q_logvar)
        eps = torch.randn_like(q_std)
        z = q_mu + eps * q_std
        
        return z, q_mu, q_logvar, pos_enc
    
    def decode(self, z, bs, n, pos_enc=None):
        if pos_enc is None:
            rand_pe = sample_unit_sphere(bs, n, self.n_max_pos) # [bs, n, n_max_pos]
            rand_pe = rand_pe.to(z.device)
            pos_enc = self.embedding_pos_enc(rand_pe)

        z = self.proj_d_model(z)
        x = z.unsqueeze(1).expand(bs, n, self.d_model)
        x = x + pos_enc

        e = z.unsqueeze(1).unsqueeze(2).expand(bs, n, n, self.d_model)
        e = sym_tensor(e)

        e = e + pos_enc.unsqueeze(1) + pos_enc.unsqueeze(2)
        e = sym_tensor(e)

        for layer in self.decoder_layers:
            x, e = layer(x, e)
        
        x = self.ln_x(x)
        x = self.proj_x(x)

        e = self.ln_e(e)
        e = self.proj_e(e)

        return x, e
        
    def forward(self, g, h, e, pos_enc, bs, n):
        if self.training:
            z, q_mu, q_logvar, pe = self.encode(g, h, e, pos_enc, bs, n)
            pe = pe.view(bs, n, self.d_model)
        else:
            z = torch.Tensor(bs, self.d_rep).normal_(mean=0.0, std=1.0).to(self.device) 
            q_mu, q_logvar, pe = None, None, None
        
        x, e = self.decode(z, bs, n, pe)

        return x, e, q_mu, q_logvar
        
    def loss(self, x_pred, e_pred, q_mu, q_logvar, x_target, e_target, bs, n):
        loss_data = self.loss_fn(x_pred.view(bs * n, self.n_atom_type), x_target)
        loss_data = loss_data + self.loss_fn(e_pred.view(bs * n * n, self.n_bond_type), e_target.view(bs * n * n))
        loss_KL = -0.15 * torch.mean( 1.0 + q_logvar - q_mu.pow(2.0) - q_logvar.exp() )
        loss_VAE = 1.0 * loss_data + loss_KL
        
        return loss_VAE
