#!/usr/bin/env python3
"""
Curate Expansive CNS Alzheimer's Drug Dataset (9 to 35 Heavy Atoms)
Sources:
  1. Master Curated BBB+ Permeable Dataset (4,947 molecules)
  2. Master Alzheimer's BACE-1 Active Dataset (1,500+ molecules)
  3. ZINC Drug-Like CNS-Filtered Subset (10,000 molecules)
Total: ~16,500 diverse drug-like molecules for training MolGAN-QPolicy and Graph VAE.
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MIN_ATOMS = 9
MAX_ATOMS = 35

def canonicalize_clean_smiles(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    if '.' in s:
        # Filter out salts / multi-components
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        n_heavy = mol.GetNumHeavyAtoms()
        if not (MIN_ATOMS <= n_heavy <= MAX_ATOMS):
            return None
        # Verify organic atoms only: C, N, O, F, S, Cl, Br, P
        allowed_nums = {6, 7, 8, 9, 15, 16, 17, 35}
        for a in mol.GetAtoms():
            if a.GetAtomicNum() not in allowed_nums:
                return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None

def main():
    print("=" * 80)
    print("BUILDING EXPANSIVE CNS ALZHEIMER'S GENERATIVE DATASET (9 - 35 HEAVY ATOMS)")
    print("=" * 80)

    curated_smiles = set()

    # 1. Master Curated BBB+ Dataset
    bbb_csv = os.path.join(PROJECT_ROOT, 'phase3_testing_models', 'bbb_sota', 'master_curated_bbb_dataset.csv')
    if os.path.exists(bbb_csv):
        df_bbb = pd.read_csv(bbb_csv)
        pos = df_bbb[df_bbb['bbb_label'] == 1]
        print(f"Loading BBB+ molecules from {bbb_csv} ({len(pos)} candidates)...")
        c_count = 0
        for s in pos['smiles']:
            can = canonicalize_clean_smiles(s)
            if can:
                curated_smiles.add(can)
                c_count += 1
        print(f"  ✓ Added {c_count} BBB+ permeable molecules (9-35 atoms).")

    # 2. Master Alzheimer's BACE-1 Dataset
    bace_csv = os.path.join(PROJECT_ROOT, 'phase3_testing_models', 'bace1_dual', 'master_bbb_bace1_dataset.csv')
    if os.path.exists(bace_csv):
        df_bace = pd.read_csv(bace_csv)
        print(f"Loading Alzheimer's BACE-1 molecules from {bace_csv} ({len(df_bace)} candidates)...")
        b_count = 0
        for s in df_bace['smiles']:
            can = canonicalize_clean_smiles(s)
            if can and can not in curated_smiles:
                curated_smiles.add(can)
                b_count += 1
        print(f"  ✓ Added {b_count} unique Alzheimer's BACE-1 target molecules.")

    # 3. ZINC 20k Drug-Like Dataset (CNS-MPO Filtered)
    zinc_csv = '/home/cse-sdpl/MolGAN-QRL-Alzheimers/data/zinc_20k.csv'
    if os.path.exists(zinc_csv):
        print(f"Screening ZINC drug-like dataset from {zinc_csv}...")
        df_zinc = pd.read_csv(zinc_csv)
        z_count = 0
        target_zinc = 10000
        for s in df_zinc['smiles']:
            can = canonicalize_clean_smiles(s)
            if can and can not in curated_smiles:
                mol = Chem.MolFromSmiles(can)
                if mol is not None:
                    # CNS filter: MW <= 450, TPSA <= 90, HBD <= 2, RotBonds <= 8
                    mw = Descriptors.MolWt(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    rotb = Lipinski.NumRotatableBonds(mol)
                    if mw <= 450 and tpsa <= 90 and hbd <= 2 and rotb <= 8:
                        curated_smiles.add(can)
                        z_count += 1
                        if z_count >= target_zinc:
                            break
        print(f"  ✓ Added {z_count} high-quality CNS drug-like molecules from ZINC.")

    print(f"\nTotal curated unique single-fragment molecules ({MIN_ATOMS}-{MAX_ATOMS} atoms): {len(curated_smiles):,}")

    # Save to SMI file
    out_smi = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35.smi')
    with open(out_smi, 'w') as f:
        for smi in sorted(curated_smiles):
            f.write(smi + '\n')
    print(f"Saved SMILES file: {out_smi}")

    # Build SparseMolecularDataset
    print("\nBuilding SparseMolecularDataset (.sparsedataset) with max vertices N=35...")
    try:
        from data.sparse_molecular_dataset import SparseMolecularDataset
        data = SparseMolecularDataset()
        data.generate(
            out_smi,
            filters=lambda m: MIN_ATOMS <= m.GetNumAtoms() <= MAX_ATOMS
        )
        out_sparse = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35.sparsedataset')
        data.save(out_sparse)
        print("--- Dataset Encoding Summary ---")
        print("  Max vertices (atoms):", data.vertexes)
        print("  Atom types (with PAD):", data.atom_num_types)
        print("  Bond types (with PAD):", data.bond_num_types)
        print("  Total encoded molecules:", len(data))
        print(f"  ✓ Successfully saved: {out_sparse}")
    except Exception as e:
        print(f"Error encoding sparse dataset: {e}")

if __name__ == '__main__':
    main()
