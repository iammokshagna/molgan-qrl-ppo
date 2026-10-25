import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DESCRIPTOR_NAMES = [
    'MW', 'TPSA', 'LogP', 'HBD', 'HBA',
    'RotBonds', 'AromaticRings', 'FractionCSP3', 'HeavyAtoms', 'QED'
]


def mol_from_input(mol_or_smiles):
    if isinstance(mol_or_smiles, str):
        return Chem.MolFromSmiles(mol_or_smiles)
    elif isinstance(mol_or_smiles, Chem.Mol):
        return mol_or_smiles
    return None


def extract_fingerprint(mol_or_smiles, n_bits=2048, radius=2):
    mol = mol_from_input(mol_or_smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def extract_physicochemical_descriptors(mol_or_smiles):
    mol = mol_from_input(mol_or_smiles)
    if mol is None:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)

    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    try:
        from rdkit.Chem import QED
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.5

    return np.array([
        mw, tpsa, logp, hbd, hba,
        rot_bonds, aromatic_rings, fraction_csp3, heavy_atoms, qed_val
    ], dtype=np.float32)


def extract_combined_features(mol_or_smiles, n_bits=2048, radius=2, scaler=None):
    fp = extract_fingerprint(mol_or_smiles, n_bits=n_bits, radius=radius)
    desc = extract_physicochemical_descriptors(mol_or_smiles)
    if scaler is not None:
        desc = scaler.transform(desc.reshape(1, -1))[0]
    return np.concatenate([fp, desc], axis=0).astype(np.float32)
