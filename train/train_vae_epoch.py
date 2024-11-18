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
        batch_h, batch_e = data_sampler.generate_batch()
        bs, n_nodes = batch_h.size()

        input_x = batch_h.to(device)
        input_e = batch_e.to(device)

        ox, oe, q_mu, q_logvar = model(input_x, input_e, bs, n_nodes)
        loss = model.loss(ox, oe, q_mu, q_logvar, input_x, input_e, bs, n_nodes)
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
