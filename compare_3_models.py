#!/usr/bin/env python3
"""
Comparative Benchmarking Suite for 3 Quantum Generative Models:
1. Model 1: MolGAN-QPolicy (Initial Model - Yukesh-Naren et al., QM9 4-Qubit PPO)
2. Model 2: MolGAN-QRL (Ihebhergli22 - 9-Qubit Value Network GAN, Large Molecules)
3. Model 3: MolGAN-QPolicy-Alzheimers (Our Current Extended Model - 9-Qubit Autoregressive PPO + VAE + Multi-Objective CNS)

Executes both:
- Small test (e.g. N = 20-25 samples)
- Complete run (e.g. N = 200-250 samples)
Computing generation metrics (Validity, Uniqueness, Novelty, Diversity, NVU),
physicochemical properties (MW, LogP, TPSA, HBD, HBA, QED, SAS),
and Alzheimer's target screening (SOTA BBB Permeability %, BACE-1 Activity %, Pfizer CNS-MPO).
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, Draw
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Model 1 imports
from ini_model.models_qpolicy import ActorRGCN_Quantum as ActorQM9
from ini_model.reward import matrices_to_mols as m2m_qm9

# Model 2 imports
from qrl_model.models_gan import Generator as GeneratorQRL
from qrl_model.sparse_molecular_dataset import SparseMolecularDataset

# Model 3 imports
from phase1_autoencoder.model import GraphVAE
from phase2_molgan_qpolicy.models_qpolicy import ActorRGCN_Quantum as ActorAlzheimers
from phase2_molgan_qpolicy.reward import matrices_to_mols as m2m_alzheimers, calculate_sas

# Phase 3 Screening evaluators
from phase3_testing_models.bbb_sota.predict_sota_bbb import predict_bbb_permeability
from phase3_testing_models.bace1_dual.predict import evaluate_molecule


def load_reference_datasets():
    """Loads reference SMILES for Novelty computation."""
    print("Loading reference datasets for Novelty calculation...")
    qm9_path = os.path.join(PROJECT_ROOT, 'data', 'qm9.smi')
    cns_path = os.path.join(PROJECT_ROOT, 'data', 'cns_alzheimers_9_35.smi')
    
    qm9_set = set()
    if os.path.exists(qm9_path):
        with open(qm9_path, 'r') as f:
            for line in f:
                s = line.strip()
                if s:
                    qm9_set.add(s)
        print(f"  ✓ Loaded QM9 reference SMILES: {len(qm9_set):,} compounds")

    cns_set = set()
    if os.path.exists(cns_path):
        with open(cns_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cns_set.add(parts[0])
        print(f"  ✓ Loaded CNS Alzheimer's reference SMILES: {len(cns_set):,} compounds")

    return qm9_set, cns_set


def compute_internal_diversity(mols):
    """Computes mean pairwise Tanimoto distance using Morgan fingerprints."""
    fps = []
    for m in mols:
        if m is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                fps.append(fp)
            except Exception:
                pass
    if len(fps) <= 1:
        return 0.0
    
    from rdkit import DataStructs
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    if not sims:
        return 0.0
    return float(1.0 - np.mean(sims))


def generate_model_1(num_samples, device, temperature=1.0):
    """
    Model 1: MolGAN-QPolicy (Initial Model - Yukesh-Naren QM9 4-Qubit PPO)
    """
    print(f"\n[1/3] Generating from Model 1: MolGAN-QPolicy (Initial 4-Qubit QM9)...")
    t0 = time.time()
    actor = ActorQM9(max_atoms=9, num_atom_types=4, num_edge_types=5, node_dim=5).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, 'ini_model', 'actor_epoch_30.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'actor_epoch_30.pt')
    actor.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    actor.eval()

    batch_chunk = 50
    raw_mols = []
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            current_b = min(batch_chunk, num_samples - generated_count)
            nodes, adjs, _, _, _ = actor.generate_mols(batch_size=current_b, device=device, temperature=temperature)
            mols_b = m2m_qm9(nodes, adjs)
            raw_mols.extend(mols_b)
            generated_count += current_b

    elapsed = time.time() - t0
    print(f"  ✓ Model 1 generation completed in {elapsed:.2f}s ({num_samples} samples, {elapsed/num_samples:.3f}s/mol)")
    return raw_mols, "Model 1: MolGAN-QPolicy (Initial QM9)"


def generate_model_2(num_samples, device):
    """
    Model 2: MolGAN-QRL (Ihebhergli22 - 9-Qubit Value Network GAN)
    """
    print(f"\n[2/3] Generating from Model 2: MolGAN-QRL (Ihebhergli22 Large Molecule GAN)...")
    t0 = time.time()
    dpath = os.path.join(PROJECT_ROOT, 'qrl_model', 'ad_bbb_35nodes.sparsedataset')
    if not os.path.exists(dpath):
        dpath = '/home/cse-sdpl/MolGAN-QRL/data/ad_bbb_35nodes.sparsedataset'

    data = SparseMolecularDataset()
    data.load(dpath)

    gen = GeneratorQRL([256, 512, 1024], 32, data.vertexes, data.bond_num_types, data.atom_num_types, 0.0).to(device)
    ckpt_path = os.path.join(PROJECT_ROOT, 'qrl_model', '150-G.ckpt')
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()

    raw_mols = []
    batch_chunk = 100
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            current_b = min(batch_chunk, num_samples - generated_count)
            z = torch.randn(current_b, 32, device=device)
            e_hat, n_hat = gen(z)
            probs_e = torch.nn.functional.softmax(e_hat, -1).cpu().numpy()
            e_hard = torch.argmax(e_hat, -1).cpu().numpy()
            n_hard = torch.argmax(n_hat, -1).cpu().numpy()
            for i in range(current_b):
                m = data.matrices2mol(n_hard[i], e_hard[i], strict=True, edge_probs=probs_e[i], min_ring_size=5)
                raw_mols.append(m)
            generated_count += current_b

    elapsed = time.time() - t0
    print(f"  ✓ Model 2 generation completed in {elapsed:.2f}s ({num_samples} samples, {elapsed/num_samples:.4f}s/mol)")
    return raw_mols, "Model 2: MolGAN-QRL (Ihebhergli22)"


def generate_model_3(num_samples, device, temperature=1.1, min_atoms=9, max_atoms=25):
    """
    Model 3: MolGAN-QPolicy-Alzheimers (Current Extended 9-Qubit Autoregressive PPO + VAE)
    """
    print(f"\n[3/3] Generating from Model 3: MolGAN-QPolicy-Alzheimers (Current Extended 9-Qubit PPO)...")
    t0 = time.time()

    # Load Phase 1 Graph VAE
    vae_path = os.path.join(PROJECT_ROOT, 'phase1_autoencoder', 'vae_model.pt')
    z_cond = None
    if os.path.exists(vae_path):
        vae = GraphVAE(vertexes=35, edges=5, nodes=10, z_dim=32).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
        z_cond = torch.randn(num_samples, 32, device=device)
    else:
        z_cond = torch.randn(num_samples, 32, device=device)

    actor = ActorAlzheimers(max_atoms=max_atoms, num_atom_types=9, node_dim=10, z_dim=32).to(device)
    ckpts = glob.glob(os.path.join(PROJECT_ROOT, 'checkpoints', 'actor_epoch_*.pt')) + \
            glob.glob(os.path.join(PROJECT_ROOT, 'Results', 'qpolicy_quantum_GAN', '*', 'model_dir', 'actor_epoch_*.pt'))
    
    # Filter for 9-qubit checkpoints
    valid_9q = []
    for c in ckpts:
        try:
            sd = torch.load(c, map_location='cpu')
            if sd['quantum_policy.weights'].shape[1] == 9:
                valid_9q.append(c)
        except Exception:
            pass
    
    if valid_9q:
        best_ckpt = max(valid_9q, key=os.path.getmtime)
        print(f"  ✓ Loading 9-qubit actor checkpoint: {best_ckpt}")
        actor.load_state_dict(torch.load(best_ckpt, map_location=device))
    else:
        print("  Notice: Using initialized 9-qubit weights.")
    actor.eval()

    raw_mols = []
    batch_chunk = 50
    generated_count = 0
    with torch.no_grad():
        while generated_count < num_samples:
            current_b = min(batch_chunk, num_samples - generated_count)
            z_b = z_cond[generated_count:generated_count + current_b] if z_cond is not None else None
            nodes, adjs, _, _, _ = actor.generate_mols(
                batch_size=current_b,
                device=device,
                temperature=temperature,
                min_atoms=min_atoms,
                max_atoms=max_atoms,
                z_cond=z_b
            )
            mols_b = m2m_alzheimers(nodes, adjs)
            raw_mols.extend(mols_b)
            generated_count += current_b

    elapsed = time.time() - t0
    print(f"  ✓ Model 3 generation completed in {elapsed:.2f}s ({num_samples} samples, {elapsed/num_samples:.3f}s/mol)")
    return raw_mols, "Model 3: MolGAN-QPolicy-Alzheimers (Ours)"


def evaluate_model_molecules(raw_mols, model_name, ref_qm9, ref_cns, device):
    """
    Evaluates generated molecules across all structural, physicochemical, and screening criteria.
    """
    total_sampled = len(raw_mols)
    valid_mols = [m for m in raw_mols if m is not None]
    validity_pct = len(valid_mols) / total_sampled * 100.0 if total_sampled > 0 else 0.0

    # Uniqueness and Canonical SMILES extraction
    records = []
    valid_connected_mols = []
    seen_canonical = set()
    unique_canonical_smiles = set()

    for m in valid_mols:
        try:
            smi = Chem.MolToSmiles(m)
            if not smi:
                continue
            # Remove stereochemistry/canonicalize
            can_smi = Chem.CanonSmiles(smi)
            is_unique = (can_smi not in seen_canonical)
            seen_canonical.add(can_smi)
            if is_unique:
                unique_canonical_smiles.add(can_smi)

            # Check disconnected components
            is_connected = ('.' not in smi)
            
            # Check novelty against both QM9 and CNS datasets
            is_novel_qm9 = can_smi not in ref_qm9
            is_novel_cns = can_smi not in ref_cns
            is_novel = is_novel_qm9 and is_novel_cns

            # Physicochemical properties
            mw = Descriptors.MolWt(m)
            clogp = Crippen.MolLogP(m)
            tpsa = Descriptors.TPSA(m)
            hbd = Descriptors.NumHDonors(m)
            hba = Descriptors.NumHAcceptors(m)
            rot_bonds = Descriptors.NumRotatableBonds(m)
            n_heavy = m.GetNumHeavyAtoms()
            qed_score = QED.qed(m)
            sas = calculate_sas(m, norm=False)

            # Lipinski Rule of 5: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
            lipinski_pass = (mw <= 500 and clogp <= 5.0 and hbd <= 5 and hba <= 10)

            rec = {
                'model': model_name,
                'smiles': smi,
                'canonical_smiles': can_smi,
                'is_connected': is_connected,
                'is_unique': is_unique,
                'is_novel': is_novel,
                'is_novel_qm9': is_novel_qm9,
                'is_novel_cns': is_novel_cns,
                'heavy_atoms': n_heavy,
                'mw': round(mw, 2),
                'clogp': round(clogp, 2),
                'tpsa': round(tpsa, 2),
                'hbd': hbd,
                'hba': hba,
                'rotatable_bonds': rot_bonds,
                'qed': round(qed_score, 4),
                'sas': round(sas, 2),
                'lipinski_pass': lipinski_pass,
                'rdkit_mol': m
            }
            records.append(rec)
            if is_connected:
                valid_connected_mols.append(m)
        except Exception:
            continue

    df_records = pd.DataFrame(records)
    unique_count = len(unique_canonical_smiles)
    uniqueness_pct = (unique_count / len(valid_mols) * 100.0) if len(valid_mols) > 0 else 0.0
    nvu_count = unique_count

    # Novelty among unique molecules
    novel_count = sum(1 for s in unique_canonical_smiles if s not in ref_qm9 and s not in ref_cns)
    novelty_pct = (novel_count / unique_count * 100.0) if unique_count > 0 else 0.0

    # Internal diversity
    diversity = compute_internal_diversity([r['rdkit_mol'] for r in records if r['is_unique']])

    # -------------------------------------------------------------------------
    # Phase 3 Target Screening (BBB Ensemble + BACE-1 Dual Predictor)
    # -------------------------------------------------------------------------
    unique_connected_records = [r for r in records if r['is_connected'] and r['is_unique']]
    screening_smiles = [r['smiles'] for r in unique_connected_records]

    bbb_probs = []
    cns_mpos = []
    bace1_probs = []
    bace1_actives = []
    tiers = []
    justifications = []

    if screening_smiles:
        # 1. SOTA BBB Multi-Representation Ensemble
        bbb_res = predict_bbb_permeability(screening_smiles)
        bbb_probs = bbb_res['ensemble_prob']
        cns_mpos = bbb_res['cns_mpo']

        # 2. Dual BACE-1 & Justifications
        for i, smi in enumerate(screening_smiles):
            dual = evaluate_molecule(smi, device=device)
            p_bace = dual['classification']['bace1_probability']
            active_bace = dual['classification']['bace1_active']
            tier = dual['classification']['tier']
            just = dual['justification']['summary_verdict']

            bace1_probs.append(p_bace)
            bace1_actives.append(active_bace)
            tiers.append(tier)
            justifications.append(just)

            # Update record
            unique_connected_records[i]['sota_bbb_prob'] = round(bbb_probs[i], 4)
            unique_connected_records[i]['sota_bbb_pass'] = (bbb_probs[i] >= 0.5)
            unique_connected_records[i]['cns_mpo'] = cns_mpos[i]
            unique_connected_records[i]['bace1_prob'] = round(p_bace, 4)
            unique_connected_records[i]['bace1_active'] = active_bace
            unique_connected_records[i]['tier'] = tier
            unique_connected_records[i]['justification'] = just

    bbb_pass_rate = (sum(1 for p in bbb_probs if p >= 0.5) / len(bbb_probs) * 100.0) if bbb_probs else 0.0
    bace1_active_rate = (sum(1 for a in bace1_actives if a) / len(bace1_actives) * 100.0) if bace1_actives else 0.0
    mean_cns_mpo = float(np.mean(cns_mpos)) if cns_mpos else 0.0
    mean_bbb_prob = float(np.mean(bbb_probs)) if bbb_probs else 0.0
    mean_bace1_prob = float(np.mean(bace1_probs)) if bace1_probs else 0.0

    # Summary metrics dictionary
    summary = {
        'model_name': model_name,
        'total_sampled': total_sampled,
        'valid_count': len(valid_mols),
        'validity_pct': round(validity_pct, 2),
        'unique_count': unique_count,
        'uniqueness_pct': round(uniqueness_pct, 2),
        'nvu': nvu_count,
        'novel_count': novel_count,
        'novelty_pct': round(novelty_pct, 2),
        'diversity': round(diversity, 4),
        'mean_heavy_atoms': round(float(df_records['heavy_atoms'].mean()), 2) if not df_records.empty else 0.0,
        'mean_mw': round(float(df_records['mw'].mean()), 2) if not df_records.empty else 0.0,
        'mean_clogp': round(float(df_records['clogp'].mean()), 2) if not df_records.empty else 0.0,
        'mean_tpsa': round(float(df_records['tpsa'].mean()), 2) if not df_records.empty else 0.0,
        'mean_qed': round(float(df_records['qed'].mean()), 4) if not df_records.empty else 0.0,
        'mean_sas': round(float(df_records['sas'].mean()), 2) if not df_records.empty else 0.0,
        'lipinski_pass_pct': round(float(df_records['lipinski_pass'].mean() * 100.0), 2) if not df_records.empty else 0.0,
        'screened_candidates_count': len(screening_smiles),
        'mean_sota_bbb_prob': round(mean_bbb_prob, 4),
        'bbb_pass_pct': round(bbb_pass_rate, 2),
        'mean_bace1_prob': round(mean_bace1_prob, 4),
        'bace1_active_pct': round(bace1_active_rate, 2),
        'mean_cns_mpo': round(mean_cns_mpo, 2)
    }

    return summary, df_records, unique_connected_records


def render_model_grid(records, output_image_path, title, max_mols=16):
    """Renders 2D grid image of candidate molecules for a single model."""
    if not records:
        return
    mols = []
    legends = []
    for i, r in enumerate(records[:max_mols]):
        m = r['rdkit_mol']
        mols.append(m)
        bbb_str = f"BBB: {r.get('sota_bbb_prob', 0.0)*100:.0f}%" if 'sota_bbb_prob' in r else f"MW: {r['mw']:.0f}"
        bace_str = f"BACE1: {r.get('bace1_prob', 0.0)*100:.0f}%" if 'bace1_prob' in r else f"QED: {r['qed']:.2f}"
        mpo_str = f"MPO: {r.get('cns_mpo', 0.0):.1f}" if 'cns_mpo' in r else f"SAS: {r['sas']:.1f}"
        legends.append(f"#{i+1}\n{bbb_str} | {bace_str}\n{mpo_str}")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(300, 300),
        legends=legends
    )
    img.save(output_image_path)
    print(f"  ✓ Saved 2D structure grid: {output_image_path}")


def render_combined_3_models_grid(rec1, rec2, rec3, output_image_path, n_per_model=4):
    """Renders a combined 3-row grid showing top candidates from all 3 models."""
    combined_mols = []
    combined_legends = []

    for name_label, rec_list in [
        ("M1 (Initial QM9)", rec1),
        ("M2 (MolGAN-QRL)", rec2),
        ("M3 (Alzheimers)", rec3)
    ]:
        for i, r in enumerate(rec_list[:n_per_model]):
            combined_mols.append(r['rdkit_mol'])
            bbb_val = r.get('sota_bbb_prob', 0.0)
            bace_val = r.get('bace1_prob', 0.0)
            mpo_val = r.get('cns_mpo', 0.0)
            combined_legends.append(
                f"[{name_label} #{i+1}]\nBBB: {bbb_val*100:.0f}% | BACE1: {bace_val*100:.0f}%\nMPO: {mpo_val:.2f} | MW: {r['mw']:.0f}"
            )

    if combined_mols:
        img = Draw.MolsToGridImage(
            combined_mols,
            molsPerRow=n_per_model,
            subImgSize=(300, 300),
            legends=combined_legends
        )
        img.save(output_image_path)
        print(f"  ✓ Saved combined 3-model comparison grid: {output_image_path}")


def run_benchmark(mode='test', num_samples=25, output_dir=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'Results', f'comparison_{mode}')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print(f"🔬 MolGAN 3-Model Comparative Benchmark [{mode.upper()} MODE: N = {num_samples} per model]")
    print(f"Device: {device} | Output Directory: {output_dir}")
    print("=" * 90)

    # 1. Load Reference Sets
    ref_qm9, ref_cns = load_reference_datasets()

    # 2. Run Generation across all 3 models
    raw_mols_1, name_1 = generate_model_1(num_samples, device)
    raw_mols_2, name_2 = generate_model_2(num_samples, device)
    raw_mols_3, name_3 = generate_model_3(num_samples, device)

    # 3. Evaluate each model
    print("\n" + "=" * 90)
    print("Evaluating Generated Molecules & Computing Cheminformatics/Screening Metrics...")
    print("=" * 90)

    summary_1, df_1, screened_1 = evaluate_model_molecules(raw_mols_1, name_1, ref_qm9, ref_cns, device)
    summary_2, df_2, screened_2 = evaluate_model_molecules(raw_mols_2, name_2, ref_qm9, ref_cns, device)
    summary_3, df_3, screened_3 = evaluate_model_molecules(raw_mols_3, name_3, ref_qm9, ref_cns, device)

    # 4. Export CSVs
    if not df_1.empty:
        df_1.drop(columns=['rdkit_mol']).to_csv(os.path.join(output_dir, 'model_1_qm9_molecules.csv'), index=False)
    if not df_2.empty:
        df_2.drop(columns=['rdkit_mol']).to_csv(os.path.join(output_dir, 'model_2_qrl_molecules.csv'), index=False)
    if not df_3.empty:
        df_3.drop(columns=['rdkit_mol']).to_csv(os.path.join(output_dir, 'model_3_alzheimers_molecules.csv'), index=False)

    # Master Comparative Summary Table
    comparison_df = pd.DataFrame([summary_1, summary_2, summary_3])
    comp_csv = os.path.join(output_dir, 'models_comparison_summary.csv')
    comparison_df.to_csv(comp_csv, index=False)
    print(f"\n✓ Exported master comparative summary CSV: {comp_csv}")

    # 5. Render 2D Chemical Grid Images
    render_model_grid(screened_1, os.path.join(output_dir, 'model_1_sample_grid.png'), name_1)
    render_model_grid(screened_2, os.path.join(output_dir, 'model_2_sample_grid.png'), name_2)
    render_model_grid(screened_3, os.path.join(output_dir, 'model_3_sample_grid.png'), name_3)
    render_combined_3_models_grid(screened_1, screened_2, screened_3, os.path.join(output_dir, 'combined_3_models_grid.png'))

    # 6. Print Formatted Direct Comparison Table
    print("\n" + "=" * 90)
    print(f"📊 BENCHMARK COMPARATIVE RESULTS SUMMARY [{mode.upper()} RUN, N = {num_samples}]")
    print("=" * 90)

    headers = [
        ("Metric", "metric"),
        ("Model 1: MolGAN-QPolicy (Initial)", summary_1),
        ("Model 2: MolGAN-QRL (Ihebhergli22)", summary_2),
        ("Model 3: MolGAN-QPolicy-Alzheimers (Ours)", summary_3)
    ]

    metric_rows = [
        ("Total Molecules Sampled", "total_sampled", ""),
        ("Decodable Valid Count", "valid_count", ""),
        ("Validity Rate (%)", "validity_pct", "%"),
        ("Unique Molecules Count", "unique_count", ""),
        ("Uniqueness Rate (%)", "uniqueness_pct", "%"),
        ("NVU (Valid & Unique)", "nvu", ""),
        ("Novelty Rate (%)", "novelty_pct", "%"),
        ("Internal Diversity", "diversity", ""),
        ("Average Heavy Atoms", "mean_heavy_atoms", ""),
        ("Average Molecular Weight (Da)", "mean_mw", " Da"),
        ("Average LogP (CLogP)", "mean_clogp", ""),
        ("Average TPSA (Å²)", "mean_tpsa", " Å²"),
        ("Drug-Likeness (QED)", "mean_qed", ""),
        ("Synthetic Accessibility (SAS)", "mean_sas", ""),
        ("Lipinski Rule-of-5 Pass (%)", "lipinski_pass_pct", "%"),
        ("Screened Candidates Count", "screened_candidates_count", ""),
        ("SOTA BBB Permeability Prob", "mean_sota_bbb_prob", ""),
        ("BBB Permeable Pass Rate (%)", "bbb_pass_pct", "%"),
        ("BACE-1 Inhibition Prob", "mean_bace1_prob", ""),
        ("BACE-1 Active Rate (%)", "bace1_active_pct", "%"),
        ("Pfizer CNS-MPO Score (/6.0)", "mean_cns_mpo", " / 6.0")
    ]

    print(f"{'Metric':<34} | {'Model 1 (Initial QM9)':<22} | {'Model 2 (MolGAN-QRL)':<22} | {'Model 3 (Alzheimers)':<22}")
    print("-" * 105)
    for name, key, unit in metric_rows:
        v1 = f"{summary_1[key]}{unit}"
        v2 = f"{summary_2[key]}{unit}"
        v3 = f"{summary_3[key]}{unit}"
        print(f"{name:<34} | {v1:<22} | {v2:<22} | {v3:<22}")
    print("=" * 105)

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="MolGAN 3-Model Comparative Benchmark")
    parser.add_argument('--mode', type=str, choices=['test', 'complete', 'both'], default='both',
                        help="Run mode: 'test' (N=20), 'complete' (N=250), or 'both' sequentially")
    parser.add_argument('--test_samples', type=int, default=20, help="Number of samples for small test (default: 20)")
    parser.add_argument('--complete_samples', type=int, default=250, help="Number of samples for complete run (default: 250)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.mode in ['test', 'both']:
        print("\n" + "#" * 90)
        print("STAGE 1: RUNNING SMALL TEST COMPARISON")
        print("#" * 90)
        run_benchmark(mode='test', num_samples=args.test_samples, seed=args.seed)

    if args.mode in ['complete', 'both']:
        print("\n" + "#" * 90)
        print("STAGE 2: RUNNING COMPLETE COMPREHENSIVE BENCHMARK")
        print("#" * 90)
        run_benchmark(mode='complete', num_samples=args.complete_samples, seed=args.seed)


if __name__ == '__main__':
    main()
