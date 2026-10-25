import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
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
from molgan_qrl.utils import MolecularMetrics


def label2onehot(labels, dim, device):
    out = torch.zeros(list(labels.size()) + [dim], device=device)
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train Graph Variational Autoencoder on Alzheimer's & BBB Dataset")
    parser.add_argument('--dataset', type=str, default='data/ad_bbb_35nodes.sparsedataset')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.005, help='Weight for KL divergence loss')
    parser.add_argument('--z_dim', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='vae')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Phase 1: Training Graph Variational Autoencoder (VAE) ===")
    print(f"Device: {device}")

    # 1. Load dataset
    dpath = os.path.join(PROJECT_ROOT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    data = SparseMolecularDataset()
    data.load(dpath)
    print(f"Loaded dataset: {dpath} (Vertices: {data.vertexes}, Atom types: {data.atom_num_types}, Bond types: {data.bond_num_types})")
    print(f"Train samples: {len(data.train_idx)}, Val samples: {len(data.validation_idx)}")

    # 2. Initialize VAE
    vae = GraphVAE(
        vertexes=data.vertexes,
        edges=data.bond_num_types,
        nodes=data.atom_num_types,
        z_dim=args.z_dim
    ).to(device)

    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Graph VAE initialized | Total trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Cross-entropy class weights
    # Bond weights: class 0 is no-bond (90% frequency), upweight actual bonds (classes 1-4)
    edge_weights = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0], device=device)
    # Node weights: downweight PAD (class 0), upweight heavy atoms
    node_weights = torch.tensor([0.5, 1.0, 2.0, 1.5, 3.0, 5.0, 3.0, 3.0, 5.0, 5.0], device=device)

    num_batches = len(data.train_idx) // args.batch_size
    best_val_loss = float('inf')
    os.makedirs(os.path.join(PROJECT_ROOT, args.save_dir), exist_ok=True)

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(args.num_epochs):
        vae.train()
        epoch_recon_e = 0.0
        epoch_recon_n = 0.0
        epoch_kl = 0.0
        epoch_loss = 0.0

        for b in range(num_batches):
            _, _, _, a_np, x_np, _, _, _, _ = data.next_train_batch(args.batch_size)
            a = torch.from_numpy(a_np).long().to(device)
            x = torch.from_numpy(x_np).long().to(device)

            a_tensor = label2onehot(a, data.bond_num_types, device)
            x_tensor = label2onehot(x, data.atom_num_types, device)

            edges_logits, nodes_logits, mu, logvar, z = vae(a_tensor, x_tensor)

            # Reconstruction loss
            loss_e = F.cross_entropy(edges_logits.reshape(-1, data.bond_num_types), a.reshape(-1), weight=edge_weights)
            loss_n = F.cross_entropy(nodes_logits.reshape(-1, data.atom_num_types), x.reshape(-1), weight=node_weights)

            # KL divergence loss
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / a.size(0)

            total_loss = loss_e + loss_n + args.beta * loss_kl

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon_e += loss_e.item()
            epoch_recon_n += loss_n.item()
            epoch_kl += loss_kl.item()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / num_batches
        avg_e = epoch_recon_e / num_batches
        avg_n = epoch_recon_n / num_batches
        avg_kl = epoch_kl / num_batches

        # 4. Validation step
        vae.eval()
        val_loss = 0.0
        val_batches = max(1, len(data.validation_idx) // args.batch_size)
        val_valid_count = 0
        val_total_count = 0

        with torch.no_grad():
            for _ in range(val_batches):
                _, _, _, a_np, x_np, _, _, _, _ = data.next_validation_batch()
                b_size = a_np.shape[0]
                a = torch.from_numpy(a_np).long().to(device)
                x = torch.from_numpy(x_np).long().to(device)

                a_tensor = label2onehot(a, data.bond_num_types, device)
                x_tensor = label2onehot(x, data.atom_num_types, device)

                edges_logits, nodes_logits, mu, logvar, z = vae(a_tensor, x_tensor)
                l_e = F.cross_entropy(edges_logits.reshape(-1, data.bond_num_types), a.reshape(-1), weight=edge_weights)
                l_n = F.cross_entropy(nodes_logits.reshape(-1, data.atom_num_types), x.reshape(-1), weight=node_weights)
                l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b_size
                val_loss += (l_e + l_n + args.beta * l_kl).item()

                # Test decoding validity
                probs_e = F.softmax(edges_logits, -1).cpu().numpy()
                e_hard = torch.argmax(edges_logits, -1).cpu().numpy()
                n_hard = torch.argmax(nodes_logits, -1).cpu().numpy()
                for i in range(b_size):
                    m = data.matrices2mol(n_hard[i], e_hard[i], strict=True, edge_probs=probs_e[i], min_ring_size=5)
                    if m is not None:
                        val_valid_count += 1
                val_total_count += b_size

        avg_val_loss = val_loss / val_batches
        val_valid_pct = (val_valid_count / max(1, val_total_count)) * 100
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.num_epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:02d}/{args.num_epochs:02d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {avg_loss:.3f} (Edges: {avg_e:.3f}, Nodes: {avg_n:.3f}, KL: {avg_kl:.2f}) | "
                  f"Val Loss: {avg_val_loss:.3f} | Recon Validity: {val_valid_pct:.1f}%", flush=True)

        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            full_model_path = os.path.join(PROJECT_ROOT, args.save_dir, 'vae_model.pt')
            torch.save(vae.state_dict(), full_model_path)

            # Export decoder state_dict directly for MolGAN Generator initialization
            decoder_ckpt_path = os.path.join(PROJECT_ROOT, args.save_dir, 'vae_decoder_for_generator.ckpt')
            torch.save(vae.decoder.state_dict(), decoder_ckpt_path)

    print(f"\nVAE Training Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Saved full VAE model to: {os.path.join(PROJECT_ROOT, args.save_dir, 'vae_model.pt')}")
    print(f"Saved Generator-ready Decoder weights to: {os.path.join(PROJECT_ROOT, args.save_dir, 'vae_decoder_for_generator.ckpt')}")


if __name__ == '__main__':
    main()
