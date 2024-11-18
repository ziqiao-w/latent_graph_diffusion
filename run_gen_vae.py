import pickle
import torch

from data_prep import sample_molecule_size
from lib.utils import Dictionary, Molecule, from_pymol_to_smile, group_molecules_per_size
from layer.dense_vae import DenseVAE
from test.test_my_vae import generate_valid_molecules


def gpu_setup():
    if torch.cuda.is_available():
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def test_gen_vae(model_type, model_config, weight_path):
    # Setup GPU device
    device = gpu_setup()

    # Load dataset
    data_folder_pytorch = 'dataset/QM9_1.4k_pytorch/'

    with open(data_folder_pytorch+"atom_dict.pkl","rb") as f:
        atom_dict=pickle.load(f)
    with open(data_folder_pytorch+"bond_dict.pkl","rb") as f:
        bond_dict=pickle.load(f)
    with open(data_folder_pytorch+"train_pytorch.pkl","rb") as f:
        train=pickle.load(f)

    num_atom_type = len(atom_dict.idx2word)
    num_bond_type = len(bond_dict.idx2word)

    train_group = group_molecules_per_size(train)
    max_mol_sz = max(list(train_group.keys()))
    
    print('Trainset is loaded, length:', len(train))
    print('Number of atom types:', num_atom_type)
    print('Number of bond types:', num_bond_type)
    print('Maximum molecule size:', max_mol_sz)

    model = model_type(**model_config)
    model = model.to(device)

    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    sampler_size = sample_molecule_size(train_group)

    perc_valid_mol, list_mol, line, list_valid_mol = generate_valid_molecules(
        model, sampler_size, atom_dict, bond_dict
    )

    print('percentage of valid molecules')

    # check the number of unique novel molecules in the generated molecules
    valid_mol_set = set(list_valid_mol)
    print('num_valid_mol: ',len(valid_mol_set))

        
    list_train_mol = [] 
    for idx in range(len(train)):
        list_train_mol.append(train[idx].smile) 

    num_unique_noval_mol = 0
    train_mol_set = set(list_train_mol)
    for mol in valid_mol_set:
        if mol not in train_mol_set:
            num_unique_noval_mol += 1

    print('num_unique_noval_mol: ',num_unique_noval_mol)

if __name__ == '__main__':
    model_type = DenseVAE
    config = {
        'd_rep': 64,
        'd_model': 256,
        'n_heads': 16,
        'n_layers': 4,
        'n_atom_type': 9, 
        'n_bond_type': 4, 
        'n_max_pos': 9, 
        'dropout': 0.0
    }
    weight_path = 'chkpt/DenseVAE_11-18-17-06.pth'
    
    test_gen_vae(model_type, config, weight_path)
