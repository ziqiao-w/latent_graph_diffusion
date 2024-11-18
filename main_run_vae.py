"""
    IMPORTING LIBS
"""
import numpy as np
import os
import time
import random
import argparse, json
from tqdm import tqdm

import torch

import torch.optim as optim

from lib.utils import MoleculeDGL, MoleculeDatasetDGL, Dictionary, Molecule

from data_prep import MoleculeSamplerDGL
from layer.vae import Transformer_VAE
from train.train_vae_epoch import train_epoch


def set_all_seeds(seed, deterministic_flag=False, tf32_flag=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = deterministic_flag
    torch.backends.cuda.matmul.allow_tf32 = tf32_flag


"""
    GPU Setup
"""
def gpu_setup():
    if torch.cuda.is_available():
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    COUNT MODEL CONFIG AND PARAMS
"""
def count_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += param.numel()
    print(f'{type(model)} model parameters: {total_param / 1e6: .2f} million')
    return total_param


"""
    TRAINING CODE
"""

def train_pipeline(model_type, model_config, dataset, params, device, out_dir):
    time_stamp = time.strftime('%m-%d-%H-%M', time.localtime())
    record_name = model_type.__name__ + '_' + time_stamp

    train_start = time.time()
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    # setting seeds
    set_all_seeds(params['seed'], tf32_flag=True)
    
    print("Trainset length: ", len(trainset))

    model = model_type(**model_config)
    model = model.to(device)

    count_model_param(model)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=params['init_lr'], 
        weight_decay=params['weight_decay']
    )

    batch_size = params['batch_size']
    num_warmup = 2 * max(20, len(trainset) // batch_size )
    print('LR scheduler num_warmup :',num_warmup)

    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda t: min((t + 1) / num_warmup, 1.0) 
    ) # warmup scheduler
    
    scheduler_tracker = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['lr_reduce_factor'],
        patience=params['lr_schedule_patience'],
        verbose=True
    )

    print("Training started")

    now_warmup = 0
    for epoch in tqdm(range(params['epochs'])):
        start = time.time()
        sampler = MoleculeSamplerDGL(trainset, batch_size)

        epoch_loss, now_warmup = train_epoch(
            model,
            optimizer,
            scheduler_warmup,
            sampler,
            now_warmup,
            num_warmup,
            device,
            params['clip_norm']
        )
        
        # Average stats
        if now_warmup >= num_warmup:
            scheduler_tracker.step(epoch_loss) # tracker scheduler defined w.r.t. loss value
            now_warmup += 1
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start) / 60
        if epoch % 5 == 0:
            print('epoch= %d \t time= %.4f min \t lr= %.7f \t loss= %.4f' % (epoch, elapsed, optimizer.param_groups[0]['lr'], epoch_loss))

        if epoch % 100 == 0:
            torch.save(model.state_dict(), out_dir + record_name +  '.pth')

        # Check lr value  
        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print("\n lr is equal to min lr -- training stopped\n")
            break

    
    # Save the model
    torch.save(model.state_dict(), out_dir + record_name +  '.pth')

    """
        Write the results in out_dir/results folder
    """
    with open(out_dir + record_name + '.txt', 'w') as f:
        f.write("""Model: {}\nparams={}\nTotal Parameters: {}\nConvergence Time (Epochs): {}\nTotal Time Taken: {:.4f} hrs\n"""\
        .format(model_type, params, count_model_param(model), epoch + 1, (time.time() - train_start) / 3600))
    





def main():    
    parser = argparse.ArgumentParser()

    # parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    # parser.add_argument('--dataset', help="Please give a value for dataset name")
    # parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', type=int, default=3, help="Please give a value for seed")
    parser.add_argument('--epochs', type=int, default=1, help="Please give a value for epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Please give a value for batch_size")
    parser.add_argument('--init_lr', type=float, default=0.0005, help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5, help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', type=int, default=10, help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Please give a value for weight_decay")
    parser.add_argument('--clip_norm', type=float, default=1.0, help="Please give a value for clip_norm")
    
    args = parser.parse_args()

    config = {
        'd_rep': 64,
        'd_model': 256,
        'n_heads': 16,
        'n_layers': 4,
        'n_atom_type': 9, 
        'n_bond_type': 4, 
        'n_max_pos': 8, 
        'dropout': 0.0
    }

    # device
    device = gpu_setup()
    
    dataset_name = 'QM9_1.4k'
    data_folder_dgl = 'dataset/QM9_1.4k_dgl_PE/'
    datasets_dgl = MoleculeDatasetDGL(dataset_name, data_folder_dgl)

    out_dir = 'chkpt/'

    train_pipeline(
        Transformer_VAE,
        config,
        datasets_dgl,
        vars(args),
        device,
        out_dir
    )

    
if __name__ == '__main__':
    main()    