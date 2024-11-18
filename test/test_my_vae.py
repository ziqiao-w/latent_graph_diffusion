import time
import torch
from tqdm import tqdm

from lib.utils import Molecule, from_pymol_to_smile
from test.utils import is_valid_generated_smiles

def generate_valid_molecules(
        model, 
        sampler_size, 
        atom_dict,
        bond_dict,
        num_gen_mol=1000, 
        num_generated_mols_per_batch=20,
    ):
    num_atom_type = len(atom_dict.idx2word)

    num_batches = num_gen_mol // num_generated_mols_per_batch 
    num_valid_mol = 0
    list_valid_mol = []
    list_gen_mol = []
    
    start = time.time()
    for idx in tqdm(range(num_batches)):
        model.eval()
        with torch.no_grad():  
            num_atom_sampled = sampler_size.choose_molecule_size() # sample the molecule size
            # num_atom_sampled = num_atom # same size
            batch_x_0, batch_e_0, _, _ =  model(g=None, h=None, e=None, pos_enc=None, bs=num_generated_mols_per_batch, n=num_atom_sampled) # [bs, n, num_atom_type], [bs, n, n, num_bond_type]
            # batch_x_0, batch_e_0, _, _  = net(0, 0, False, num_generated_mols_per_batch, num_atom_sampled) # [bs, n, num_atom_type], [bs, n, n, num_bond_type]
            batch_x_0 = torch.max(batch_x_0,dim=2)[1]  # [bs, n] 
            batch_e_0 = torch.max(batch_e_0,dim=3)[1]  # [bs, n, n]

            x_hat = batch_x_0.detach().to('cpu')
            e_hat = batch_e_0.detach().to('cpu')
            
            for x, e in zip(x_hat, e_hat):
                pymol = Molecule(num_atom_sampled, num_atom_type)
                pymol.atom_type = x
                pymol.bond_type = e
                smile = from_pymol_to_smile(pymol, atom_dict, bond_dict)
                list_gen_mol.append(smile)
                
                # mol = Chem.MolFromSmiles(smile)
                if is_valid_generated_smiles(smile, min_atoms=num_atom_sampled, max_atoms=num_atom_sampled):
                    list_valid_mol.append(smile)
                    num_valid_mol += 1


    perc_valid_molecules = 100 * num_valid_mol / num_gen_mol
    line = '\t num_gen_mol= ' + str(num_gen_mol) + '   time(min)= ' + str((time.time()-start)/60)[:5] + '   perc valid molecules= ' + str(perc_valid_molecules)[:6]
    
    return perc_valid_molecules, list_gen_mol, line, list_valid_mol
