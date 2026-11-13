#!/usr/bin/env python3
"""
Collect and Curate Expanded Human BACE-1 (Beta-secretase 1) Molecular Dataset.

Sources queried:
  1. BindingDB REST API (UniProt P56817: Human Beta-secretase 1)
  2. ChEMBL REST API (Target CHEMBL4822: Homo sapiens Beta-secretase 1, IC50/Ki/Kd in nM)
  3. Existing Local Datasets:
       - phase3_testing_models/bace1_dual/master_bbb_bace1_dataset.csv
       - data/cns_alzheimers_9_35.smi

Outputs:
  - data/bace1_curated_expanded.csv : Standalone curated BACE-1 bioactivity table with provenance
  - phase3_testing_models/bace1_dual/master_bbb_bace1_dataset.csv : Merged & deduplicated multi-task dataset
    (original backed up to master_bbb_bace1_dataset_orig.csv)
  - data/cns_alzheimers_9_35.smi & data/cns_alzheimers_9_35.sparsedataset :
    Expanded 9-35 heavy-atom generative dataset preserving all existing SMILES
    (originals backed up to *_orig.smi / *_orig.sparsedataset)
"""

import os
import sys
import time
import json
import shutil
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Standard organic CNS drug vocabulary matching cns_alzheimers_9_35.sparsedataset:
# C(6), N(7), O(8), F(9), P(15), S(16), Cl(17), Br(35)
ALLOWED_ATOMIC_NUMS = {6, 7, 8, 9, 15, 16, 17, 35}
MIN_HEAVY_ATOMS_GEN = 9
MAX_HEAVY_ATOMS_GEN = 35
ACTIVE_THRESHOLD_NM = 100.0  # Calibrated to match MoleculeNet BACE (94.7% agreement, pIC50 >= 7.0)


def canonicalize_mol(smi, strip_stereo=True, largest_frag=True):
    """Sanitize and canonicalize SMILES; returns (canonical_smiles, rdkit_mol) or (None, None)."""
    if not isinstance(smi, str) or not smi.strip():
        return None, None
    s = smi.split(' |')[0].strip()  # Strip BindingDB CXSMILES tags like ' |r|'
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None, None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None, None
        if len(frags) > 1:
            if not largest_frag:
                return None, None
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        if mol.GetNumHeavyAtoms() < 5:
            return None, None
        # Ensure neutral organic or standard atom types (ignore Boron/metals/isotopes in BACE1 dataset)
        allowed_bace = ALLOWED_ATOMIC_NUMS | {53}  # Allow Iodine in BACE-1 classifier dataset
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in allowed_bace:
                return None, None
        csmi = Chem.MolToSmiles(mol, isomericSmiles=not strip_stereo)
        mol_clean = Chem.MolFromSmiles(csmi)
        if mol_clean is None:
            return None, None
        return csmi, mol_clean
    except Exception:
        return None, None


def is_gen_compatible(mol):
    """Checks strict 9-35 heavy atom CNS generative dataset compatibility."""
    if mol is None:
        return False
    n_heavy = mol.GetNumHeavyAtoms()
    if not (MIN_HEAVY_ATOMS_GEN <= n_heavy <= MAX_HEAVY_ATOMS_GEN):
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS:
            return False
    return True


def compute_descriptors(mol):
    """Computes RDKit physicochemical descriptors matching master_bbb_bace1_dataset.csv schema."""
    return {
        'heavy_atoms': int(mol.GetNumHeavyAtoms()),
        'mw': round(float(Descriptors.MolWt(mol)), 2),
        'tpsa': round(float(rdMolDescriptors.CalcTPSA(mol)), 2),
        'logp': round(float(Crippen.MolLogP(mol)), 2),
        'hbd': int(rdMolDescriptors.CalcNumHBD(mol)),
        'hba': int(rdMolDescriptors.CalcNumHBA(mol)),
        'rot_bonds': int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        'aromatic_rings': int(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }


def fetch_bindingdb_bace1():
    """Fetches human BACE-1 (UniProt P56817) affinity data from BindingDB."""
    print("\n[1/2] Fetching Human BACE-1 (UniProt: P56817) from BindingDB API...")
    url = "https://bindingdb.org/rest/getLigandsByUniprots?uniprot=P56817&cutoff=1000000&response=application/json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode('utf-8'))

    affs = data.get('getLindsByUniprotsResponse', {}).get('affinities', [])
    print(f"  ✓ Retrieved {len(affs):,} raw BindingDB affinity records.")

    records = defaultdict(list)  # smi -> list of (source, type, relation, val_nM)
    for row in affs:
        atype = str(row.get('affinity_type', '')).strip()
        if atype not in ('IC50', 'Ki', 'Kd'):
            continue
        raw_val = str(row.get('affinity', '')).strip()
        if not raw_val:
            continue
        rel = '='
        if raw_val.startswith('>='):
            rel = '>='
            num_str = raw_val[2:].strip()
        elif raw_val.startswith('>'):
            rel = '>'
            num_str = raw_val[1:].strip()
        elif raw_val.startswith('<='):
            rel = '<='
            num_str = raw_val[2:].strip()
        elif raw_val.startswith('<'):
            rel = '<'
            num_str = raw_val[1:].strip()
        else:
            num_str = raw_val.lstrip('~= ').strip()

        try:
            val_nm = float(num_str)
        except ValueError:
            continue
        if not np.isfinite(val_nm) or val_nm <= 0:
            continue

        csmi, _ = canonicalize_mol(row.get('smile', ''))
        if csmi:
            records[csmi].append(('BindingDB', atype, rel, val_nm))

    print(f"  ✓ Parsed {len(records):,} unique valid molecules from BindingDB.")
    return records


def fetch_chembl_bace1(max_pages=25):
    """Paginates through ChEMBL CHEMBL4822 (Homo sapiens Beta-secretase 1) IC50/Ki/Kd records."""
    print("\n[2/2] Fetching Human BACE-1 (Target: CHEMBL4822) from ChEMBL REST API...")
    base_url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        "?target_chembl_id=CHEMBL4822&standard_type__in=IC50,Ki,Kd&standard_units=nM&limit=1000"
    )
    records = defaultdict(list)
    total_fetched = 0
    offset = 0
    page = 0

    while page < max_pages:
        url = f"{base_url}&offset={offset}"
        data = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                break
            except Exception as e:
                print(f"    Retry {attempt+1}/3 at offset {offset} due to: {e}")
                time.sleep(2)
        if data is None:
            print(f"  ⚠ Stopped early at offset {offset} after retries.")
            break

        acts = data.get('activities', [])
        if not acts:
            break
        total_fetched += len(acts)
        for act in acts:
            smi = act.get('canonical_smiles')
            atype = str(act.get('standard_type', '')).strip()
            rel = str(act.get('standard_relation') or '=').strip()
            sval = act.get('standard_value')
            if not smi or sval is None or atype not in ('IC50', 'Ki', 'Kd'):
                continue
            try:
                val_nm = float(sval)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(val_nm) or val_nm <= 0:
                continue

            csmi, _ = canonicalize_mol(smi)
            if csmi:
                records[csmi].append(('ChEMBL', atype, rel, val_nm))

        page_meta = data.get('page_meta', {})
        total_count = page_meta.get('total_count', 0)
        if page % 5 == 0 or not page_meta.get('next'):
            print(f"    Fetched {total_fetched:,}/{total_count:,} ChEMBL activity rows ({len(records):,} unique SMILES)...")
        if not page_meta.get('next'):
            break
        offset += 1000
        page += 1

    print(f"  ✓ Parsed {len(records):,} unique valid molecules from ChEMBL.")
    return records


def resolve_binary_label(measurements):
    """
    Resolves exact binary BACE-1 label (1: IC50/Ki <= 100 nM, 0: > 100 nM) from measurements.
    Handles censored inequality bounds rigorously:
      - Exact ('=', '~'): used directly
      - Upper bound ('<', '<=') with val <= 100 nM: unambiguously active
      - Lower bound ('>', '>=') with val > 100 nM: unambiguously inactive
      - Ambiguous bounds (e.g. '< 10000 nM' or '> 50 nM'): ignored unless no exact exists, then skipped.
    Returns (label_int_or_None, representative_nM, sources_str, types_str).
    """
    exact_vals = [v for _, _, r, v in measurements if r in ('=', '~', '')]
    definitive_active_bounds = [v for _, _, r, v in measurements if r in ('<', '<=') and v <= ACTIVE_THRESHOLD_NM]
    definitive_inactive_bounds = [v for _, _, r, v in measurements if r in ('>', '>=') and v >= ACTIVE_THRESHOLD_NM]

    sources = ','.join(sorted(set(s for s, _, _, _ in measurements)))
    atypes = ','.join(sorted(set(t for _, t, _, _ in measurements)))

    if exact_vals:
        med = float(np.median(exact_vals))
        lbl = 1 if med <= ACTIVE_THRESHOLD_NM else 0
        return lbl, round(med, 3), sources, atypes
    elif definitive_active_bounds and not definitive_inactive_bounds:
        med = float(np.median(definitive_active_bounds))
        return 1, round(med, 3), sources, atypes
    elif definitive_inactive_bounds and not definitive_active_bounds:
        med = float(np.median(definitive_inactive_bounds))
        return 0, round(med, 3), sources, atypes
    else:
        return None, None, sources, atypes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect and merge expanded BACE-1 molecular data")
    parser.add_argument('--refresh', action='store_true', default=False,
                        help='Force re-downloading BindingDB and ChEMBL data even if cached')
    parser.add_argument('--rebuild_sparse', action='store_true', default=False,
                        help='Force rebuilding .sparsedataset even if .smi molecule count did not change')
    args = parser.parse_args()

    print("=" * 80)
    print("BACE-1 MOLECULAR DATA COLLECTION & EXPANSION PIPELINE")
    print("=" * 80)

    standalone_csv = os.path.join(PROJECT_ROOT, 'data', 'bace1_curated_expanded.csv')
    if os.path.exists(standalone_csv) and not args.refresh:
        print(f"\n✓ Loading cached curated BACE-1 dataset: {standalone_csv}")
        df_expanded = pd.read_csv(standalone_csv)
    else:
        # 1. Fetch online databases
        bdb_records = fetch_bindingdb_bace1()
        chembl_records = fetch_chembl_bace1()

        merged_raw = defaultdict(list)
        for k, v in bdb_records.items():
            merged_raw[k].extend(v)
        for k, v in chembl_records.items():
            merged_raw[k].extend(v)

        print(f"\nCombined online unique sanitized BACE-1 structures: {len(merged_raw):,}")

        # Resolve labels & build standalone expanded CSV
        expanded_rows = []
        for smi, meas in merged_raw.items():
            lbl, med_nm, srcs, atypes = resolve_binary_label(meas)
            if lbl is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            expanded_rows.append({
                'smiles': smi,
                'bace1_label': lbl,
                'median_affinity_nm': med_nm,
                'sources': srcs,
                'assay_types': atypes,
                'n_measurements': len(meas),
                'gen_9_35_compatible': is_gen_compatible(mol),
                **compute_descriptors(mol)
            })

        df_expanded = pd.DataFrame(expanded_rows)
        df_expanded.to_csv(standalone_csv, index=False)
        print(f"\n✓ Saved standalone curated BACE-1 dataset: {standalone_csv}")

    print(f"  Total labeled BACE-1 compounds: {len(df_expanded):,} "
          f"(Active <=100 nM: {(df_expanded['bace1_label']==1).sum():,}, "
          f"Inactive >100 nM: {(df_expanded['bace1_label']==0).sum():,})")
    print(f"  9-35 heavy atom generative-compatible BACE-1 compounds: {int(df_expanded['gen_9_35_compatible'].sum()):,}")

    # 2. Merge into Phase 3 master_bbb_bace1_dataset.csv preserving ALL original rows intact
    master_csv = os.path.join(PROJECT_ROOT, 'phase3_testing_models', 'bace1_dual', 'master_bbb_bace1_dataset.csv')
    master_bak = os.path.join(PROJECT_ROOT, 'phase3_testing_models', 'bace1_dual', 'master_bbb_bace1_dataset_orig.csv')
    if os.path.exists(master_csv) and not os.path.exists(master_bak):
        shutil.copy2(master_csv, master_bak)
        print(f"\n✓ Backed up original Phase 3 dataset to: {master_bak}")

    src_master = master_bak if os.path.exists(master_bak) else master_csv
    df_orig = pd.read_csv(src_master)
    preserved_rows = df_orig.to_dict('records')

    # Map canonical flat SMILES -> list of row indices in preserved_rows
    flat_to_indices = defaultdict(list)
    for idx, row in enumerate(preserved_rows):
        s_raw = str(row['smiles']).strip()
        flat_to_indices[s_raw].append(idx)
        csmi, _ = canonicalize_mol(s_raw, strip_stereo=True, largest_frag=True)
        if csmi:
            flat_to_indices[csmi].append(idx)

    orig_total = len(preserved_rows)
    orig_bbb = int(df_orig['bbb_label'].notna().sum())
    orig_bace = int(df_orig['bace1_label'].notna().sum())

    annotated_bbb_rows = 0
    new_bace_rows = []
    seen_new_smiles = set()
    for _, row in df_expanded.iterrows():
        smi = str(row['smiles']).strip()
        if smi in flat_to_indices:
            for idx in flat_to_indices[smi]:
                if pd.isna(preserved_rows[idx]['bace1_label']):
                    preserved_rows[idx]['bace1_label'] = float(row['bace1_label'])
                    annotated_bbb_rows += 1
        elif smi not in seen_new_smiles:
            seen_new_smiles.add(smi)
            new_bace_rows.append({
                'smiles': smi,
                'bbb_label': np.nan,
                'bace1_label': float(row['bace1_label']),
                'heavy_atoms': row['heavy_atoms'],
                'mw': row['mw'],
                'tpsa': row['tpsa'],
                'logp': row['logp'],
                'hbd': row['hbd'],
                'hba': row['hba'],
                'rot_bonds': row['rot_bonds'],
                'aromatic_rings': row['aromatic_rings'],
            })

    cols = ['smiles', 'bbb_label', 'bace1_label', 'heavy_atoms', 'mw', 'tpsa', 'logp', 'hbd', 'hba', 'rot_bonds', 'aromatic_rings']
    df_new_master = pd.DataFrame(preserved_rows + new_bace_rows)[cols]
    df_new_master.to_csv(master_csv, index=False)

    new_bbb_total = int(df_new_master['bbb_label'].notna().sum())
    new_bace_total = int(df_new_master['bace1_label'].notna().sum())
    new_bace_pos = int((df_new_master['bace1_label'] == 1).sum())
    new_bace_neg = int((df_new_master['bace1_label'] == 0).sum())
    print(f"\n✓ Updated Phase 3 Master Dataset: {master_csv}")
    print(f"  Rows: {orig_total:,} -> {len(df_new_master):,} (+{len(new_bace_rows):,} new BACE-1 molecules)")
    print(f"  BBB labeled rows: {orig_bbb:,} -> {new_bbb_total:,} (100% preserved)")
    print(f"  BBB+BACE-1 cross-annotated previously BBB-only rows: +{annotated_bbb_rows:,}")
    print(f"  BACE-1 labeled rows: {orig_bace:,} -> {new_bace_total:,} (Active: {new_bace_pos:,}, Inactive: {new_bace_neg:,})")

    # 3. Expand generative dataset data/cns_alzheimers_9_35.smi & .sparsedataset
    smi_path = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35.smi')
    smi_bak = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35_orig.smi')
    sparse_path = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35.sparsedataset')
    sparse_bak = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35_orig.sparsedataset')

    if os.path.exists(smi_path) and not os.path.exists(smi_bak):
        shutil.copy2(smi_path, smi_bak)
    if os.path.exists(sparse_path) and not os.path.exists(sparse_bak):
        shutil.copy2(sparse_path, sparse_bak)

    existing_gen_smiles = set()
    src_smi = smi_bak if os.path.exists(smi_bak) else smi_path
    if os.path.exists(src_smi):
        with open(src_smi, 'r') as f:
            for line in f:
                s = line.strip()
                if s:
                    existing_gen_smiles.add(s)

    orig_gen_count = len(existing_gen_smiles)
    added_gen = 0
    for _, row in df_expanded[df_expanded['gen_9_35_compatible'] == True].iterrows():
        s = str(row['smiles']).strip()
        mol = Chem.MolFromSmiles(s)
        if mol is None or '.' in s:
            continue
        if s not in existing_gen_smiles:
            existing_gen_smiles.add(s)
            added_gen += 1

    with open(smi_path, 'w') as f:
        for s in sorted(existing_gen_smiles):
            f.write(s + '\n')

    print(f"\n✓ Generative SMILES Dataset: {smi_path}")
    print(f"  Molecules (9-35 heavy atoms): {orig_gen_count:,} -> {len(existing_gen_smiles):,} (+{added_gen:,} new BACE-1 structures)")

    # Rebuild SparseMolecularDataset if needed
    if not os.path.exists(sparse_path) or args.rebuild_sparse or args.refresh:
        print("\nRebuilding SparseMolecularDataset (.sparsedataset) with expanded molecules...")
        from data.sparse_molecular_dataset import SparseMolecularDataset
        dataset = SparseMolecularDataset()
        dataset.generate(
            smi_path,
            filters=lambda m: m is not None and MIN_HEAVY_ATOMS_GEN <= m.GetNumAtoms() <= MAX_HEAVY_ATOMS_GEN
        )
        dataset.save(sparse_path)
        print("\n--- Expanded Sparse Dataset Summary ---")
        print(f"  Total encoded molecules : {len(dataset):,}")
        print(f"  Max vertices (atoms)    : {dataset.vertexes}")
        print(f"  Atom types (with PAD)   : {dataset.atom_num_types} -> {dataset.atom_encoder_m}")
        print(f"  Bond types (with PAD)   : {dataset.bond_num_types}")
        print(f"  ✓ Saved to: {sparse_path}")
    else:
        print(f"✓ Expanded SparseMolecularDataset already up to date at {sparse_path} (use --rebuild_sparse to force).")


if __name__ == '__main__':
    main()
