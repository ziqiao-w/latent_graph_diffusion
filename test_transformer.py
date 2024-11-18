import torch
import torch.nn as nn
import math
import numpy as np

from lib.utils import MoleculeDGL, MoleculeDatasetDGL, Dictionary, Molecule
from data_prep import MoleculeSamplerDGL

import pickle
from layer.transformer import GraphTransformerLayer
from layer.dense import DenseTransformerLayer
from layer.vae import Transformer_VAE


def test_sparse_model():
    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device= torch.device("cuda:0") # use GPU
    else:
        device= torch.device("cpu")
    print(device)


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

    np.random.seed(64)
    
    bs = 16
    sampler = MoleculeSamplerDGL(trainset, bs)
    print('sampler.num_mol :',sampler.num_mol)
    while(not sampler.is_empty()):
        num_batches_remaining = sampler.compute_num_batches_remaining()
        print('num_batches_remaining :',num_batches_remaining)
        batched_graph, batched_node, batched_edge = sampler.generate_batch()
        bs = batched_graph.batch_size
        n_nodes = batched_graph.num_nodes() // bs
        print('batch_size :',bs)
        print('n_nodes :',n_nodes)

        print('batched_graph :',batched_graph)
        print('batched_label :',batched_node.shape)
        print('batched_edge :',batched_edge.shape)
        atom_type = batched_graph.ndata['feat']
        print("Atom type shape:", atom_type.shape)

        bond_type = batched_graph.edata['feat']
        print("Bond type shape:", bond_type.shape)

        poc_enc  = batched_graph.ndata['pos_enc']
        print("Positional encoding shape:", poc_enc.shape)

        break

    batched_graph = batched_graph.to(device)
    batch_x = batched_graph.ndata['feat'].to(device)
    batch_e = batched_graph.edata['feat'].to(device)
    poc_enc = batched_graph.ndata['pos_enc'].to(device)

    x_target = batched_node.to(device)
    e_target = batched_edge.to(device)


    # model = GraphTransformerLayer(
    #     in_dim=64,
    #     out_dim=64,
    #     num_heads=1
    # )

    model = Transformer_VAE(
        d_rep=32,
        d_model=64,
        n_heads=2,
        n_layers=1,
        n_atom_type=num_atom_type,
        n_bond_type=num_bond_type,
        n_max_pos=8,
    )

    model = model.to(device)
    # print(model)
    
    ox, oe, q_mu, q_logvar = model(batched_graph, batch_x, batch_e, poc_enc, bs, n_nodes)
    print('ox :',ox.shape)
    print('oe :',oe.shape)
    print('q_mu :',q_mu.shape)
    print('q_logvar :',q_logvar.shape)

    loss = model.loss(ox, oe, q_mu, q_logvar, batch_x, e_target, bs, n_nodes)
    print(loss)

    loss.backward()


def test_dense_model():
    print(torch.__version__)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device= torch.device("cuda:0") # use GPU
    else:
        device= torch.device("cpu")
    print(device)

    bs, n, d = 3, 4, 64 
    h = torch.randn(bs, n, d).to(device)
    e = torch.randn(bs, n, n, d).to(device)

    model = DenseTransformerLayer(
        d_model=d,
        n_heads=2,
    ).to(device)

    h, e = model(h, e)
    print(h.shape, e.shape)


if __name__ == '__main__':
    import time
    from layer.vae import Transformer_VAE
    this_type = Transformer_VAE
    time_stamp = time.strftime('%m-%d,%H:%M', time.localtime())
    print(this_type.__name__)
    print(str(this_type) + time_stamp)
