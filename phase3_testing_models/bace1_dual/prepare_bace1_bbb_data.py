import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'alzheimers_data')
OUT_DIR = os.path.join(PROJECT_ROOT, 'phase3_bbb_bace1')
os.makedirs(OUT_DIR, exist_ok=True)


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def compute_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'mw': round(Descriptors.MolWt(mol), 2),
        'tpsa': round(rdMolDescriptors.CalcTPSA(mol), 2),
        'logp': round(Crippen.MolLogP(mol), 2),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'rot_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol)
    }


def main():
    print("=== Phase 3: Preparing Focused BBB & BACE-1 Inhibitor Dataset ===")
    
    # 1. BACE-1 Inhibition Data (bace_raw.csv)
    bace_path = os.path.join(DATA_DIR, 'bace_raw.csv')
    df_bace = pd.read_csv(bace_path)
    bace_dict = {}
    for _, row in df_bace.iterrows():
        csmi = canonicalize(row['mol'])
        if csmi:
            bace_dict[csmi] = int(row['Class'])
    print(f"Loaded BACE-1 molecules: {len(bace_dict)} (Active Inhibitors: {sum(v==1 for v in bace_dict.values())}, Inactive: {sum(v==0 for v in bace_dict.values())})")

    # Add clinical BACE-1 inhibitors
    known_path = os.path.join(DATA_DIR, 'ad_known_therapeutics.csv')
    if os.path.exists(known_path):
        df_known = pd.read_csv(known_path)
        bace_drugs = {'Verubecestat', 'Lanabecestat', 'Atabecestat', 'Umibecestat'}
        for _, row in df_known.iterrows():
            if row['name'] in bace_drugs:
                csmi = canonicalize(row['smiles'])
                if csmi:
                    bace_dict[csmi] = 1

    # 2. Blood-Brain Barrier (BBB) Data
    bbb_path = os.path.join(DATA_DIR, 'unified_bbb_dataset.csv')
    df_bbb = pd.read_csv(bbb_path)
    bbb_dict = {}
    for _, row in df_bbb.iterrows():
        csmi = canonicalize(row['smiles'])
        if csmi:
            bbb_dict[csmi] = int(row['bbb_label'])
    print(f"Loaded BBB molecules: {len(bbb_dict)} (BBB+: {sum(v==1 for v in bbb_dict.values())}, BBB-: {sum(v==0 for v in bbb_dict.values())})")

    # 3. Combine into Unified Table
    all_smiles = set(bace_dict.keys()) | set(bbb_dict.keys())
    master_records = []

    for smi in all_smiles:
        desc = compute_descriptors(smi)
        if desc is None:
            continue
        rec = {
            'smiles': smi,
            'bbb_label': bbb_dict.get(smi, np.nan),
            'bace1_label': bace_dict.get(smi, np.nan),
            **desc
        }
        master_records.append(rec)

    df_out = pd.DataFrame(master_records)
    out_file = os.path.join(OUT_DIR, 'master_bbb_bace1_dataset.csv')
    df_out.to_csv(out_file, index=False)
    print(f"\nSaved master BBB & BACE-1 dataset to: {out_file}")
    print(f"Total unique molecules: {len(df_out)}")
    print(f"BBB annotated: {df_out['bbb_label'].notna().sum()} (BBB+: {int((df_out['bbb_label']==1).sum())})")
    print(f"BACE-1 annotated: {df_out['bace1_label'].notna().sum()} (Active Inhibitors: {int((df_out['bace1_label']==1).sum())})")


if __name__ == '__main__':
    main()
