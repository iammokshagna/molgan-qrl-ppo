import os
import sys
import argparse
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PKG_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from molgan_vqp.solver_vqp import SolverJerbiVQP


def parse_args():
    parser = argparse.ArgumentParser(description='MolGAN-VQP: Jerbi et al. (NeurIPS 2021) Variational Quantum Policy Generator')

    default_dataset = os.path.join(_ROOT_DIR, 'data', 'cns_alzheimers_9_35.sparsedataset')
    if not os.path.exists(default_dataset):
        default_dataset = os.path.join(_ROOT_DIR, 'data', 'qm9_with_negatives.sparsedataset')

    default_save_dir = os.path.join(_ROOT_DIR, 'Results', 'vqp_quantum_GAN')
    default_vae_path = os.path.join(_ROOT_DIR, 'phase1_autoencoder', 'vae_model.pt')

    parser.add_argument('--dataset_path', type=str, default=default_dataset,
                        help='Path to molecular dataset (default: data/cns_alzheimers_9_35.sparsedataset or qm9)')
    parser.add_argument('--save_dir', type=str, default=default_save_dir,
                        help='Directory to save model checkpoints and images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for rollouts and training')
    parser.add_argument('--min_atoms', type=int, default=9,
                        help='Minimum heavy atom count for dynamic generation (default: 9)')
    parser.add_argument('--max_atoms', type=int, default=35,
                        help='Maximum heavy atom count for dynamic generation (default: 35)')

    # Phase 1 Graph VAE Latent Prior Conditioning
    parser.add_argument('--no_vae_prior', action='store_true', default=False,
                        help='Disable conditioning generator with Phase 1 VAE latent codes')
    parser.add_argument('--vae_model_path', type=str, default=default_vae_path,
                        help='Path to trained Phase 1 Graph VAE checkpoint')

    # Epochs & Rollouts
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO update epochs per rollout')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Number of batch steps per epoch (default: 100 = 1,600 mols/epoch; set 0 for full dataset)')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.10, help='Entropy regularization coefficient')
    parser.add_argument('--temperature', type=float, default=1.2, help='Sampling temperature for policy diversity')

    # Composite Reward weights: R = alpha * R_target + beta * sigma(D) + gamma * R_secondary
    parser.add_argument('--alpha', type=float, default=0.45, help='Weight for primary target reward (e.g. BACE-1)')
    parser.add_argument('--beta', type=float, default=0.10, help='Weight for Discriminator realness')
    parser.add_argument('--gamma', type=float, default=0.30, help='Weight for secondary target reward (e.g. BBB)')
    parser.add_argument('--prop_metric', type=str, default='bace1', choices=['bace1', 'qed', 'sas', 'np', 'dc', 'logp'],
                        help='Target chemical property metric')
    parser.add_argument('--scaffold_bucket_size', type=int, default=25,
                        help='REINVENT Scaffold Diversity Memory bucket size to prevent mode collapse')

    # Checkpoint resumption
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to existing Actor checkpoint to warm-start or resume training')

    # Learning rates
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Learning rate for Actor policy')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for Value Critic')
    parser.add_argument('--lr_disc', type=float, default=1e-4, help='Learning rate for Discriminator')

    # Policy heads (Jerbi Softmax-PQC with Multi-Layer Data Re-Uploading)
    parser.add_argument('--atom_policy', type=str, default='jerbi',
                        choices=['jerbi', 'dedicated_quantum', 'dedicated'],
                        help='Policy architecture for heavy atom selection (default: jerbi 9-qubit PQC)')
    parser.add_argument('--edge_policy', type=str, default='jerbi',
                        choices=['jerbi', 'dedicated', 'dedicated_quantum', '4q', 'quantum', 'quantum_2q', '2q', 'classical', 'mlp'],
                        help='Policy architecture for bond formation without aromatic qubit (default: jerbi 4-qubit PQC)')
    parser.add_argument('--n_layers_atom', type=int, default=3, help='Number of variational quantum layers in atom PQC')
    parser.add_argument('--n_layers_edge', type=int, default=3, help='Number of variational quantum layers in edge PQC')
    parser.add_argument('--output_scale', type=float, default=3.0, help='Learnable Softmax temperature scaling beta')

    # Masking and filtering controls
    parser.add_argument('--no_valence_mask', action='store_true', default=False,
                        help='Disable physical valence masking on bond selection')
    parser.add_argument('--no_scaffold_memory', action='store_true', default=False,
                        help='Disable REINVENT Bemis-Murcko scaffold memory penalty in reward')
    parser.add_argument('--no_tanimoto', action='store_true', default=False,
                        help='Disable in-batch Tanimoto diversity penalty in reward')
    parser.add_argument('--no_filter', action='store_true', default=False,
                        help='Disable scaffold memory and chemical alert filters in reward')
    parser.add_argument('--no_connectivity_enforce', action='store_true', default=False,
                        help='Disable post-hoc connectivity enforcement fallback')
    parser.add_argument('--no_atom_filters', action='store_true', default=False,
                        help='Disable step 0 Carbon anchor and heteroatom cap filters')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print(" " * 5 + "MolGAN-VQP: Jerbi et al. (NeurIPS 2021) Variational Quantum Policy Generator")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"  {k:24s}: {v}")
    print("=" * 80)

    steps_pe = args.steps_per_epoch if args.steps_per_epoch > 0 else None

    solver = SolverJerbiVQP(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        use_vae_prior=not args.no_vae_prior,
        vae_model_path=args.vae_model_path,
        steps_per_epoch=steps_pe,
        resume_checkpoint=args.resume_checkpoint,
        epochs=args.epochs,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        prop_metric=args.prop_metric,
        scaffold_bucket_size=args.scaffold_bucket_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_disc=args.lr_disc,
        atom_policy_type=args.atom_policy,
        edge_policy_type=args.edge_policy,
        n_layers_atom=args.n_layers_atom,
        n_layers_edge=args.n_layers_edge,
        output_scale=args.output_scale,
        use_valence_mask=not args.no_valence_mask,
        use_filter=not args.no_filter,
        use_scaffold_memory=not args.no_scaffold_memory,
        use_tanimoto=not args.no_tanimoto,
        enforce_connectivity=not args.no_connectivity_enforce,
        use_atom_filters=not args.no_atom_filters,
        device=args.device
    )

    solver.train()


if __name__ == '__main__':
    main()
