#!/usr/bin/env python3
"""
Comparative Benchmarking Suite for QM9 De Novo Molecular Generation:
1. Model 1: Classical MolGAN (Cao & Kipf, 2018 - QM9 checkpoint)
2. Model 2: MolGAN-QRL (Hergli & Harigua-Souiai, 2024 - 9-Qubit Quantum RL Critic GAN)
3. Model 3: MolGAN-QPolicy (Yukesh-Naren et al., 2026 - Dedicated 4-Qubit Element-Wise Policy Head + PPO)

Strictly evaluates on QM9 dataset without Phase 3 screening:
- Chemical Validity (Strict & Post-Processed)
- Uniqueness
- Novelty (vs QM9 reference database)
- Internal Diversity (Tanimoto distance on Morgan fingerprints)
- Number of Valid & Unique (NVU)
- Fundamental physicochemical properties (MW, LogP, TPSA, QED, SAS)
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, Draw
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Model 1 & 2 Generator and Dataset imports
from qrl_model.models_gan import Generator as GeneratorGAN
from qrl_model.sparse_molecular_dataset import SparseMolecularDataset

# Model 3 imports
from ini_model.models_qpolicy import ActorRGCN_Quantum as ActorQPolicy
from ini_model.reward import matrices_to_mols as m2m_qpolicy, calculate_sas


def load_qm9_reference():
    """Loads QM9 canonical SMILES for Novelty computation."""
    qm9_path = os.path.join(PROJECT_ROOT, 'data', 'qm9.smi')
    qm9_set = set()
    if os.path.exists(qm9_path):
        with open(qm9_path, 'r') as f:
            for line in f:
                s = line.strip().split()[0]
                if s:
                    qm9_set.add(s)
        print(f"Loaded {len(qm9_set):,} reference QM9 SMILES for novelty check.")
    else:
        print(f"Warning: {qm9_path} not found. Novelty will be measured relatively.")
    return qm9_set


def decode_strict_qm9(node_labels, edge_labels, atom_decoder, bond_decoder):
    """
    Direct RDKit reconstruction without any heuristic valence correction.
    Strictly checks whether the raw graph passes Chem.SanitizeMol.
    """
    mol = Chem.RWMol()
    node_map = {}
    for idx, nl in enumerate(node_labels):
        atomic_num = atom_decoder.get(int(nl), 0)
        if atomic_num > 0:
            m_idx = mol.AddAtom(Chem.Atom(atomic_num))
            node_map[idx] = m_idx
    
    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end and start in node_map and end in node_map:
            b_type = int(edge_labels[start, end])
            if b_type > 0 and b_type in bond_decoder:
                mol.AddBond(node_map[start], node_map[end], bond_decoder[b_type])
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def compute_internal_diversity(mols):
    """Computes internal diversity: 1 - mean pairwise Tanimoto similarity."""
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
    return float(1.0 - np.mean(sims)) if sims else 0.0


def evaluate_generated_molecules(raw_mols, dataset_qm9, qm9_ref_set, model_name, sample_size):
    """
    Computes standard QM9 metrics for a generated list of molecules.
    """
    valid_mols = [m for m in raw_mols if m is not None]
    valid_count = len(valid_mols)
    validity_pct = (valid_count / sample_size) * 100.0

    valid_smiles = []
    clean_valid_mols = []
    for m in valid_mols:
        try:
            smi = Chem.MolToSmiles(m, canonical=True)
            if smi:
                valid_smiles.append(smi)
                clean_valid_mols.append(m)
        except Exception:
            pass

    unique_smiles = list(set(valid_smiles))
    unique_count = len(unique_smiles)
    uniqueness_pct = (unique_count / (len(valid_smiles) or 1)) * 100.0
    nvu = unique_count

    # Novelty: unique molecules not in QM9 reference set
    novel_count = sum(1 for smi in unique_smiles if smi not in qm9_ref_set)
    novelty_pct = (novel_count / (unique_count or 1)) * 100.0 if unique_count > 0 else 0.0

    # Internal Diversity
    diversity = compute_internal_diversity(clean_valid_mols)

    # Core physicochemical properties
    mws, logps, tpsas, qeds, sass, heavy_atoms = [], [], [], [], [], []
    for smi in unique_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mws.append(Descriptors.MolWt(m))
            logps.append(Crippen.MolLogP(m))
            tpsas.append(Descriptors.TPSA(m))
            qeds.append(QED.qed(m))
            try:
                sass.append(calculate_sas(m, norm=False))
            except Exception:
                sass.append(5.0)
            heavy_atoms.append(m.GetNumHeavyAtoms())

    return {
        "Model": model_name,
        "Total Sampled": sample_size,
        "Valid Count": valid_count,
        "Validity Rate (%)": round(validity_pct, 2),
        "Unique Molecules": unique_count,
        "Uniqueness Rate (%)": round(uniqueness_pct, 2),
        "NVU (Valid & Unique)": nvu,
        "Novelty Rate (%)": round(novelty_pct, 2),
        "Internal Diversity": round(diversity, 4),
        "Mean Heavy Atoms": round(float(np.mean(heavy_atoms)), 2) if heavy_atoms else 0.0,
        "Mean MW (Da)": round(float(np.mean(mws)), 1) if mws else 0.0,
        "Mean LogP": round(float(np.mean(logps)), 2) if logps else 0.0,
        "Mean TPSA (Å²)": round(float(np.mean(tpsas)), 1) if tpsas else 0.0,
        "Mean QED": round(float(np.mean(qeds)), 4) if qeds else 0.0,
        "Mean SAS": round(float(np.mean(sass)), 2) if sass else 0.0,
        "smiles_list": unique_smiles,
        "valid_mols": clean_valid_mols
    }


def generate_molgan_classical(num_samples, dataset_qm9, device, seed=42):
    """Model 1: Classical MolGAN on QM9."""
    print(f"\n[1/3] Generating from Model 1: Classical MolGAN (QM9)...")
    t0 = time.time()
    ckpt_path = '/home/cse-sdpl/MolGAN-QRL/Results/GAN/2026-10-11_18-41-35/model_dir/199-G.ckpt'
    gen = GeneratorGAN([128, 256, 512], 8, 9, 5, 5, 0.0).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()

    torch.manual_seed(seed)
    raw_mols_strict = []
    raw_mols_corrected = []

    batch_chunk = 100
    generated = 0
    with torch.no_grad():
        while generated < num_samples:
            b_size = min(batch_chunk, num_samples - generated)
            z = torch.randn(b_size, 8, device=device)
            e_logits, n_logits = gen(z)
            e_hard = F.gumbel_softmax(e_logits, hard=True).argmax(dim=-1).cpu().numpy()
            n_hard = F.gumbel_softmax(n_logits, hard=True).argmax(dim=-1).cpu().numpy()

            for n, e in zip(n_hard, e_hard):
                m_strict = decode_strict_qm9(n, e, dataset_qm9.atom_decoder_m, dataset_qm9.bond_decoder_m)
                raw_mols_strict.append(m_strict)
                m_corr = dataset_qm9.matrices2mol(n, e, strict=True)
                raw_mols_corrected.append(m_corr)
            generated += b_size

    elapsed = time.time() - t0
    print(f"  ✓ Classical MolGAN completed in {elapsed:.2f}s ({elapsed/num_samples:.4f}s/mol)")
    return raw_mols_strict, raw_mols_corrected, elapsed


def generate_molgan_qrl(num_samples, dataset_qm9, device, seed=42):
    """Model 2: MolGAN-QRL on QM9."""
    print(f"\n[2/3] Generating from Model 2: MolGAN-QRL (Quantum RL Critic on QM9)...")
    t0 = time.time()
    ckpt_path = '/home/cse-sdpl/MolGAN-QRL/Results/quantum_GAN/2026-10-08_17-13-06/model_dir/1-G.ckpt'
    gen = GeneratorGAN([128, 256, 512], 8, 9, 5, 5, 0.0).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()

    torch.manual_seed(seed)
    raw_mols_strict = []
    raw_mols_corrected = []

    batch_chunk = 100
    generated = 0
    with torch.no_grad():
        while generated < num_samples:
            b_size = min(batch_chunk, num_samples - generated)
            z = torch.randn(b_size, 8, device=device)
            e_logits, n_logits = gen(z)
            e_hard = F.gumbel_softmax(e_logits, hard=True).argmax(dim=-1).cpu().numpy()
            n_hard = F.gumbel_softmax(n_logits, hard=True).argmax(dim=-1).cpu().numpy()

            for n, e in zip(n_hard, e_hard):
                m_strict = decode_strict_qm9(n, e, dataset_qm9.atom_decoder_m, dataset_qm9.bond_decoder_m)
                raw_mols_strict.append(m_strict)
                m_corr = dataset_qm9.matrices2mol(n, e, strict=True)
                raw_mols_corrected.append(m_corr)
            generated += b_size

    elapsed = time.time() - t0
    print(f"  ✓ MolGAN-QRL completed in {elapsed:.2f}s ({elapsed/num_samples:.4f}s/mol)")
    return raw_mols_strict, raw_mols_corrected, elapsed


def generate_molgan_qpolicy(num_samples, device, seed=42):
    """Model 3: MolGAN-QPolicy from Yukesh-Naren repo (Dedicated 4-Qubit Quantum Policy Head + PPO)."""
    print(f"\n[3/3] Generating from Model 3: MolGAN-QPolicy (Yukesh-Naren QM9 4-Qubit PPO)...")
    t0 = time.time()
    ckpt_path = os.path.join(PROJECT_ROOT, 'ini_model', 'actor_epoch_30.pt')
    actor = ActorQPolicy(max_atoms=9, num_atom_types=4, num_edge_types=5, node_dim=5).to(device)
    actor.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    actor.eval()

    torch.manual_seed(seed)
    raw_mols = []
    batch_chunk = 50
    generated = 0

    with torch.no_grad():
        while generated < num_samples:
            b_size = min(batch_chunk, num_samples - generated)
            nodes, adjs, _, _, _ = actor.generate_mols(batch_size=b_size, device=device, temperature=1.0)
            mols = m2m_qpolicy(nodes, adjs)
            raw_mols.extend(mols)
            generated += b_size

    elapsed = time.time() - t0
    print(f"  ✓ MolGAN-QPolicy completed in {elapsed:.2f}s ({elapsed/num_samples:.4f}s/mol)")
    return raw_mols, elapsed


def save_comparison_grid(results, out_img_path, max_mols_per_model=4):
    """Creates a side-by-side 2D chemical structure image comparison."""
    mols_to_draw = []
    legends = []
    for r in results:
        model_name = r["Model"]
        valid_mols = r["valid_mols"]
        count = 0
        for m in valid_mols:
            if m is not None:
                mols_to_draw.append(m)
                smi = Chem.MolToSmiles(m)
                legends.append(f"{model_name}\n{smi[:22]}")
                count += 1
                if count >= max_mols_per_model:
                    break

    if mols_to_draw:
        img = Draw.MolsToGridImage(
            mols_to_draw,
            molsPerRow=max_mols_per_model,
            subImgSize=(250, 200),
            legends=legends,
            useSVG=False
        )
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        img.save(out_img_path)
        print(f"Saved structure comparison grid to: {out_img_path}")


def run_benchmark(mode="test", num_samples=30, device="cuda"):
    """Executes the comparative benchmark."""
    print("=" * 80)
    print(f" QM9 3-MODEL BENCHMARK: MolGAN vs MolGAN-QRL vs MolGAN-QPolicy")
    print(f" Mode: {mode.upper()} | Sample size per model: {num_samples} | Device: {device}")
    print("=" * 80)

    # 1. Load reference QM9 data
    dataset_qm9 = SparseMolecularDataset()
    dataset_qm9.load('/home/cse-sdpl/MolGAN-QRL/data/qm9_sample.sparsedataset')
    qm9_ref_set = load_qm9_reference()

    # 2. Run Generations
    m1_strict, m1_corr, t1 = generate_molgan_classical(num_samples, dataset_qm9, device)
    m2_strict, m2_corr, t2 = generate_molgan_qrl(num_samples, dataset_qm9, device)
    m3_strict, t3 = generate_molgan_qpolicy(num_samples, device)

    # 3. Evaluate Metrics
    # (A) Strict evaluation (direct valence / sanitization check)
    r1_strict = evaluate_generated_molecules(m1_strict, dataset_qm9, qm9_ref_set, "MolGAN (Classical)", num_samples)
    r1_strict["Generation Time (s)"] = round(t1, 2)
    r1_strict["Speed (s/mol)"] = round(t1 / num_samples, 4)

    r2_strict = evaluate_generated_molecules(m2_strict, dataset_qm9, qm9_ref_set, "MolGAN-QRL", num_samples)
    r2_strict["Generation Time (s)"] = round(t2, 2)
    r2_strict["Speed (s/mol)"] = round(t2 / num_samples, 4)

    r3_strict = evaluate_generated_molecules(m3_strict, dataset_qm9, qm9_ref_set, "MolGAN-QPolicy", num_samples)
    r3_strict["Generation Time (s)"] = round(t3, 2)
    r3_strict["Speed (s/mol)"] = round(t3 / num_samples, 4)

    # (B) Corrected evaluation for MolGAN & MolGAN-QRL
    r1_corr = evaluate_generated_molecules(m1_corr, dataset_qm9, qm9_ref_set, "MolGAN (with repair)", num_samples)
    r1_corr["Generation Time (s)"] = round(t1, 2)
    r1_corr["Speed (s/mol)"] = round(t1 / num_samples, 4)

    r2_corr = evaluate_generated_molecules(m2_corr, dataset_qm9, qm9_ref_set, "MolGAN-QRL (with repair)", num_samples)
    r2_corr["Generation Time (s)"] = round(t2, 2)
    r2_corr["Speed (s/mol)"] = round(t2 / num_samples, 4)

    # 4. Compile Tables
    summary_cols = [
        "Model", "Total Sampled", "Valid Count", "Validity Rate (%)",
        "Unique Molecules", "Uniqueness Rate (%)", "NVU (Valid & Unique)",
        "Novelty Rate (%)", "Internal Diversity", "Mean Heavy Atoms",
        "Mean MW (Da)", "Mean LogP", "Mean TPSA (Å²)", "Mean QED", "Mean SAS",
        "Speed (s/mol)"
    ]

    df_strict = pd.DataFrame([r1_strict, r2_strict, r3_strict])[summary_cols]
    df_corrected = pd.DataFrame([r1_corr, r2_corr, r3_strict])[summary_cols]

    # Save Results
    out_dir = os.path.join(PROJECT_ROOT, "Results", f"comparison_qm9_{mode}")
    os.makedirs(out_dir, exist_ok=True)
    df_strict.to_csv(os.path.join(out_dir, "strict_benchmark_summary.csv"), index=False)
    df_corrected.to_csv(os.path.join(out_dir, "corrected_benchmark_summary.csv"), index=False)

    # Save SMILES
    with open(os.path.join(out_dir, "molgan_smiles.txt"), "w") as f:
        f.write("\n".join(r1_strict["smiles_list"]))
    with open(os.path.join(out_dir, "molgan_qrl_smiles.txt"), "w") as f:
        f.write("\n".join(r2_strict["smiles_list"]))
    with open(os.path.join(out_dir, "molgan_qpolicy_smiles.txt"), "w") as f:
        f.write("\n".join(r3_strict["smiles_list"]))

    # Save Comparison Grid Image
    img_path_ws = os.path.join(out_dir, f"{mode}_qm9_structures.png")
    img_path_artifact = f"/home/cse-sdpl/.gemini/antigravity/brain/3bcadab4-aeee-4ee0-b28f-ccc02883ff64/images/{mode}_qm9_structures.png"
    save_comparison_grid([r1_strict, r2_strict, r3_strict], img_path_ws, max_mols_per_model=4)
    save_comparison_grid([r1_strict, r2_strict, r3_strict], img_path_artifact, max_mols_per_model=4)

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: {mode.upper()} RUN (STRICT RDKit SANITIZATION)")
    print("=" * 80)
    print(df_strict.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: {mode.upper()} RUN (WITH STANDARD VALENCE REPAIR)")
    print("=" * 80)
    print(df_corrected.to_string(index=False))

    return df_strict, df_corrected, r1_strict, r2_strict, r3_strict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QM9 Benchmark Comparison: MolGAN vs MolGAN-QRL vs MolGAN-QPolicy")
    parser.add_argument("--mode", type=str, choices=["test", "complete", "custom"], default="test",
                        help="Benchmark mode: test (N=30) or complete (N=500)")
    parser.add_argument("--samples", type=int, default=None, help="Custom sample count per model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Execution device (cuda or cpu)")
    args = parser.parse_args()

    if args.mode == "test":
        n_samples = args.samples or 30
    elif args.mode == "complete":
        n_samples = args.samples or 500
    else:
        n_samples = args.samples or 100

    run_benchmark(mode=args.mode, num_samples=n_samples, device=args.device)
