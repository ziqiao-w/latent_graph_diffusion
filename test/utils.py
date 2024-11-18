from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')


def is_reasonable_atom_count(mol, min_atoms=3, max_atoms=50):
    num_atoms = mol.GetNumAtoms()
    return min_atoms <= num_atoms <= max_atoms


def is_valid_generated_smiles(smiles, min_atoms=3, max_atoms=50):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except:
        return False
    return is_reasonable_atom_count(mol, min_atoms, max_atoms)
