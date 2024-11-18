# Libraries
import dgl
import torch

import numpy as np
import math

from lib.utils import Dictionary, Molecule, from_pymol_to_smile


def group_molecules_per_size(dataset):
    mydict={}
    for mol in dataset:
        if len(mol) not in mydict:
            mydict[len(mol)]=[]
        mydict[len(mol)].append(mol)
    return mydict


def group_molecules_per_size_dgl(dataset):
    mydict={}
    for mol in dataset:
        g = mol[0]
        n = g.number_of_nodes()
        
        if n not in mydict:
            mydict[n]=[]
        
        mydict[n].append(mol)
    return mydict


class MoleculeSampler:
    def __init__(self, organized_dataset, bs , shuffle=True):  
        self.bs = bs
        self.num_mol =  { sz: len(list_of_mol)  for sz , list_of_mol in organized_dataset.items() }
        self.counter = { sz: 0   for sz in organized_dataset }
        if shuffle:
            self.order = { sz: np.random.permutation(num)  for sz , num in self.num_mol.items() }
        else:
            self.order = { sz: np.arange(num)  for sz , num in self.num_mol.items() } 

    def compute_num_batches_remaining(self):
        return {sz:  math.ceil(((self.num_mol[sz] - self.counter[sz])/self.bs))  for sz in self.num_mol} 

    def choose_molecule_size(self):
        num_batches = self.compute_num_batches_remaining()
        possible_sizes =  np.array( list( num_batches.keys()) )
        prob           =  np.array( list( num_batches.values() )   ) 
        prob =  prob / prob.sum()
        sz   = np.random.choice(  possible_sizes , p=prob )
        return sz

    def is_empty(self):
        num_batches= self.compute_num_batches_remaining()
        return sum( num_batches.values() ) == 0

    def draw_batch_of_molecules(self, sz):  
        num_batches = self.compute_num_batches_remaining()
        if (self.num_mol[sz] - self.counter[sz])/self.bs >= 1.0:
            bs = self.bs
        else:
            bs = self.num_mol[sz] - (self.num_mol[sz]//self.bs) * self.bs
        #print('sz, bs',sz, bs)
        indices = self.order[sz][ self.counter[sz] : self.counter[sz] + bs]
        self.counter[sz] += bs 
        return indices
    

class MoleculeSamplerDGL(MoleculeSampler):

    def __init__(self, dgl_dataset, bs, shuffle=True):
        self.data_group = group_molecules_per_size_dgl(dgl_dataset)
        super().__init__(self.data_group, bs, shuffle)

    def generate_batch(self):
        sz = self.choose_molecule_size()
        indices = self.draw_batch_of_molecules(sz)
        samples = [self.data_group[sz][i] for i in indices]
        graphs, labels = map(list, zip(*samples))

        batch_size = len(graphs)
        batched_edge = torch.zeros(batch_size, sz, sz, dtype=torch.long)
        
        for i, g in enumerate(graphs):
            u, v = g.all_edges(order='eid')
            weight_adj = batched_edge[i]
            weight_adj[u, v] = g.edata['feat']

        batched_graph = dgl.batch(graphs)
        batched_label = torch.stack(labels)
        return batched_graph, batched_label, batched_edge
    

class MoleculeSamplerPytorch(MoleculeSampler):
    def __init__(self, dgl_dataset, bs, shuffle=True):
        self.data_group = group_molecules_per_size(dgl_dataset)
        super().__init__(self.data_group, bs, shuffle)

    def generate_batch(self):
        sz = self.choose_molecule_size()
        indices = self.draw_batch_of_molecules(sz)
        samples = [self.data_group[sz][i] for i in indices]

        batch_h = torch.stack([s.atom_type for s in samples]).long()
        batch_e = torch.stack([s.bond_type for s in samples]).long()

        return batch_h, batch_e
    

class sample_molecule_size:

    def __init__(self, organized_dataset):  
        self.num_mol =  { sz: len(list_of_mol)  for sz , list_of_mol in organized_dataset.items() }
        self.num_batches_remaining = { sz:  self.num_mol[sz]  for sz in self.num_mol } 
    
    def choose_molecule_size(self):
        num_batches = self.num_batches_remaining
        possible_sizes =  np.array( list( num_batches.keys()) )
        prob           =  np.array( list( num_batches.values() )   ) 
        prob =  prob / prob.sum()
        sz   = np.random.choice(  possible_sizes , p=prob )
        return sz
