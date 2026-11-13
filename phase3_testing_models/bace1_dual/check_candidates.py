import os
import sys
import glob
import json
import argparse
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, 'molgan_qrl'),
    CUR_DIR
])

from molgan_qrl.models_gan import Generator
from data.sparse_molecular_dataset import SparseMolecularDataset
from predict import evaluate_molecule


def find_latest_checkpoint(results_dir):
    pattern = os.path.join(results_dir, '**', '*-G.ckpt')
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Screen & Justify Generated Candidates strictly for BBB Permeability and BACE-1 Inhibition")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to generator checkpoint (*-G.ckpt)')
    parser.add_argument('--dataset', type=str, default='data/ad_bbb_35nodes.sparsedataset')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--min_atoms', type=int, default=25)
    parser.add_argument('--max_atoms', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='Results/phase3_bbb_bace1_candidates')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Phase 3: Focused Blood-Brain Barrier & BACE-1 Screening Pipeline ===")
    print(f"Device: {device}")

    # 1. Load dataset
    dpath = os.path.join(PROJECT_ROOT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    data = SparseMolecularDataset()
    data.load(dpath)

    # 2. Initialize Generator
    G = Generator([256, 512, 1024], 32, data.vertexes, data.bond_num_types, data.atom_num_types, 0.0).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(os.path.join(PROJECT_ROOT, 'Results'))

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading Generator weights from: {ckpt_path}")
        G.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("Notice: No trained checkpoint specified. Sampling from initialized Generator.")

    G.eval()

    # 3. Sample molecules
    print(f"\nSampling {args.num_samples} candidates (batch size {args.batch_size})...")
    all_mols = []
    sampled = 0
    with torch.no_grad():
        while sampled < args.num_samples:
            curr_b = min(args.batch_size, args.num_samples - sampled)
            z = torch.randn(curr_b, 32, device=device)
            e_hat, n_hat = G(z)
            probs_e = torch.nn.functional.softmax(e_hat, -1).cpu().numpy()
            e_hard = torch.argmax(e_hat, -1).cpu().numpy()
            n_hard = torch.argmax(n_hat, -1).cpu().numpy()

            for i in range(curr_b):
                m = data.matrices2mol(n_hard[i], e_hard[i], strict=True, edge_probs=probs_e[i], min_ring_size=5)
                all_mols.append(m)
            sampled += curr_b

    # 4. Filter valid & unique
    valid_candidates = []
    seen = set()
    for m in all_mols:
        if m is not None:
            smi = Chem.MolToSmiles(m)
            if smi and smi not in seen and '.' not in smi:
                seen.add(smi)
                valid_candidates.append(m)

    size_candidates = [m for m in valid_candidates if args.min_atoms <= m.GetNumHeavyAtoms() <= args.max_atoms]
    print(f"Sampling Summary:")
    print(f"  Total generated: {len(all_mols)}")
    print(f"  Valid non-disconnected: {len(valid_candidates)} ({len(valid_candidates)/len(all_mols)*100:.1f}%)")
    print(f"  Large molecules ({args.min_atoms}-{args.max_atoms} heavy atoms): {len(size_candidates)}")

    target_pool = size_candidates if len(size_candidates) >= 5 else valid_candidates
    if not target_pool:
        print("Warning: No valid molecules generated in this sampling run.")
        return

    # 5. Run Phase 3 Evaluation strictly for BBB and BACE-1
    print("\nRunning Phase 3 Screening (strictly BBB Permeability & BACE-1 Inhibition)...")
    eval_records = []
    dossier = []

    for idx, mol in enumerate(target_pool):
        smi = Chem.MolToSmiles(mol)
        rep = evaluate_molecule(smi, device=device)
        dossier.append(rep)

        cns_props = rep['cns_properties']
        cls_info = rep['classification']
        just_info = rep['justification']
        qed_val = round(float(QED.qed(mol)), 3)

        rec = {
            'candidate_id': idx + 1,
            'smiles': smi,
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'mw': cns_props['mw'],
            'tpsa': cns_props['tpsa'],
            'logp': cns_props['clogp'],
            'hbd': cns_props['hbd'],
            'hba': cns_props['hba'],
            'rot_bonds': cns_props['rotatable_bonds'],
            'qed': qed_val,
            'cns_mpo_score': cns_props['cns_mpo_score'],
            'bbb_permeable': cls_info['bbb_permeable'],
            'bbb_probability': cls_info['bbb_probability'],
            'bace1_active': cls_info['bace1_active'],
            'bace1_probability': cls_info['bace1_probability'],
            'classification_tier': cls_info['tier'],
            'summary_verdict': just_info['summary_verdict'],
            'bbb_justification': "; ".join(just_info['bbb_permeability_reasons']),
            'bace1_justification': "; ".join(just_info['bace1_inhibition_reasons'])
        }
        eval_records.append(rec)

    df_eval = pd.DataFrame(eval_records)
    df_eval['composite_score'] = df_eval['bbb_probability'] + df_eval['bace1_probability'] + (df_eval['cns_mpo_score'] / 6.0)
    df_eval.sort_values(by=['composite_score'], ascending=False, inplace=True)

    # 6. Save Outputs
    out_dir = os.path.join(PROJECT_ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'generated_bbb_bace1_candidates.csv')
    df_eval.to_csv(csv_path, index=False)
    print(f"\nSaved Candidate Evaluation Table to: {csv_path}")

    json_path = os.path.join(out_dir, 'bbb_bace1_justification_dossier.json')
    with open(json_path, 'w') as f:
        json.dump(dossier, f, indent=2)
    print(f"Saved Full Justification Dossier to: {json_path}")

    # 7. Render 2D image grid of top candidates
    top_candidates = df_eval.head(6)
    mols_to_draw = []
    legends = []

    for _, row in top_candidates.iterrows():
        m = Chem.MolFromSmiles(row['smiles'])
        if m:
            mols_to_draw.append(m)
            legend = (
                f"ID {row['candidate_id']} | Atoms: {row['heavy_atoms']}\n"
                f"BBB: {row['bbb_probability']*100:.1f}% | BACE-1: {row['bace1_probability']*100:.1f}%\n"
                f"CNS-MPO: {row['cns_mpo_score']}/6.0 | QED: {row['qed']:.2f}\n"
                f"{row['classification_tier'].split('(')[0].strip()}"
            )
            legends.append(legend)

    if mols_to_draw:
        img_path = os.path.join(out_dir, 'top_bbb_bace1_candidates.png')
        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=3, subImgSize=(350, 320), legends=legends)
        img.save(img_path)
        print(f"Saved 2D Chemical Structure Grid to: {img_path}")

    # 8. Print Display Table
    print("\n================== TOP GENERATED BBB & BACE-1 INHIBITOR CANDIDATES ==================")
    for rank, (_, row) in enumerate(top_candidates.iterrows(), 1):
        print(f"\nRank {rank} [Candidate #{row['candidate_id']}]:")
        print(f"  SMILES: {row['smiles']}")
        print(f"  Heavy Atoms: {row['heavy_atoms']} | MW: {row['mw']:.1f} Da | TPSA: {row['tpsa']:.1f} Å² | LogP: {row['logp']:.2f}")
        print(f"  CNS-MPO Score: {row['cns_mpo_score']}/6.0 | QED Druglikeness: {row['qed']:.3f}")
        print(f"  Blood-Brain Barrier (BBB): {'PASS (Permeable)' if row['bbb_permeable'] else 'FAIL (Impermeable)'} (Confidence: {row['bbb_probability']*100:.1f}%)")
        print(f"  BACE-1 Inhibition: {'ACTIVE INHIBITOR' if row['bace1_active'] else 'INACTIVE'} (Confidence: {row['bace1_probability']*100:.1f}%)")
        print(f"  Tier: {row['classification_tier']}")
        print(f"  Verdict: {row['summary_verdict']}")
        print(f"  BBB Analysis: {row['bbb_justification']}")
        print(f"  BACE-1 Analysis: {row['bace1_justification']}")
    print("======================================================================================\n")


if __name__ == '__main__':
    main()
