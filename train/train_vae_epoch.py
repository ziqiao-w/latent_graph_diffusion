"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math


def train_epoch(
        model, 
        optimizer, 
        lr_scheduler,
        data_sampler, 
        acc_warmup,
        max_warmup,
        device, 
        clip_norm=1.0
    ):
    
    model.train()
    epoch_loss = 0
    n_steps = 0
    optimizer.zero_grad()

    while not data_sampler.is_empty():
        batched_graph, batched_node, batched_edge = data_sampler.generate_batch()
        bs = batched_graph.batch_size
        n_nodes = batched_graph.num_nodes() // bs

        graphs = batched_graph.to(device)
        input_x = graphs.ndata['feat'].to(device)
        input_e = graphs.edata['feat'].to(device)
        pos_enc = graphs.ndata['pos_enc'].to(device)

        sign_flip = torch.rand(pos_enc.size(1)).to(device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        pos_enc = pos_enc * sign_flip.unsqueeze(0)

        batch_e = batched_edge.to(device)

        ox, oe, q_mu, q_logvar = model(graphs, input_x, input_e, pos_enc, bs, n_nodes)
        loss = model.loss(ox, oe, q_mu, q_logvar, input_x, batch_e, bs, n_nodes)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad()

        if acc_warmup < max_warmup:
            lr_scheduler.step()
        
        acc_warmup += 1

        epoch_loss += loss.detach().item()
        n_steps += 1

    epoch_loss /= n_steps

    return epoch_loss, acc_warmup
