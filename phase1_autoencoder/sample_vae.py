import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, 'molgan_qrl'),
    os.path.join(PROJECT_ROOT, 'vae')
])

from data.sparse_molecular_dataset import SparseMolecularDataset
from vae.model import GraphVAE


def main():
    parser = argparse.ArgumentParser(description="Sample and Interpolate in VAE Latent Space")
    parser.add_argument('--model_path', type=str, default='vae/vae_model.pt')
    parser.add_argument('--dataset', type=str, default='data/ad_bbb_35nodes.sparsedataset')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='Results/vae_samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Sampling from Graph Variational Autoencoder Latent Space ===")

    # Load dataset
    dpath = os.path.join(PROJECT_ROOT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    data = SparseMolecularDataset()
    data.load(dpath)

    # Load VAE
    mpath = os.path.join(PROJECT_ROOT, args.model_path) if not os.path.isabs(args.model_path) else args.model_path
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Trained VAE model not found at {mpath}. Run train_vae.py first.")

    vae = GraphVAE(vertexes=data.vertexes, edges=data.bond_num_types, nodes=data.atom_num_types, z_dim=32).to(device)
    vae.load_state_dict(torch.load(mpath, map_location=device))
    vae.eval()
    print(f"Loaded trained VAE from: {mpath}")

    # 1. Random Prior Sampling: z ~ N(0, I)
    z = torch.randn(args.num_samples, 32, device=device)
    with torch.no_grad():
        edges_logits, nodes_logits = vae.decode(z)
        probs_e = F.softmax(edges_logits, -1).cpu().numpy()
        edges_hard = torch.argmax(edges_logits, -1).cpu().numpy()
        nodes_hard = torch.argmax(nodes_logits, -1).cpu().numpy()

    valid_mols = []
    for i in range(args.num_samples):
        m = data.matrices2mol(nodes_hard[i], edges_hard[i], strict=True, edge_probs=probs_e[i], min_ring_size=5)
        if m is not None:
            valid_mols.append(m)

    validity_pct = (len(valid_mols) / args.num_samples) * 100
    print(f"\nPrior Sampling Results:")
    print(f"  Sampled: {args.num_samples}")
    print(f"  Valid Molecules: {len(valid_mols)} ({validity_pct:.1f}%)")

    # Save images
    os.makedirs(os.path.join(PROJECT_ROOT, args.out_dir), exist_ok=True)
    if valid_mols:
        img_path = os.path.join(PROJECT_ROOT, args.out_dir, 'vae_prior_samples.png')
        display_mols = valid_mols[:6]
        legends = [f"Atoms: {m.GetNumHeavyAtoms()}, Bonds: {m.GetNumBonds()}" for m in display_mols]
        img = Draw.MolsToGridImage(display_mols, molsPerRow=3, subImgSize=(350, 320), legends=legends)
        img.save(img_path)
        print(f"Saved 2D chemical structure image grid to: {img_path}")


if __name__ == '__main__':
    main()
