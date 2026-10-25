import os
import sys
import argparse
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PKG_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from phase2_molgan_qpolicy.solver_qpolicy import SolverQPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='MolGAN-QRL with Dedicated 1-Qubit = 1-Atom Policy Head')

    default_dataset = os.path.join(_ROOT_DIR, 'data', 'cns_alzheimers_9_35.sparsedataset')
    default_save_dir = os.path.join(_ROOT_DIR, 'Results', 'qpolicy_quantum_GAN')

    parser.add_argument('--dataset_path', type=str, default=default_dataset,
                        help='Path to the preprocessed CNS Alzheimer dataset')
    parser.add_argument('--save_dir', type=str, default=default_save_dir,
                        help='Directory to save model checkpoints and images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for rollouts and training')
    parser.add_argument('--min_atoms', type=int, default=9, help='Minimum heavy atoms for generated candidates')
    parser.add_argument('--max_atoms', type=int, default=35, help='Maximum heavy atoms for generated candidates')
    parser.add_argument('--use_vae_prior', action='store_true', default=True, help='Condition generator with Phase 1 VAE latent codes')
    parser.add_argument('--vae_model_path', type=str, default=os.path.join(_ROOT_DIR, 'phase1_autoencoder', 'vae_model.pt'),
                        help='Path to pre-trained Phase 1 Graph VAE checkpoint')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO update epochs per rollout')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.10, help='Entropy regularization coefficient')
    parser.add_argument('--temperature', type=float, default=1.2, help='Sampling temperature for policy diversity')

    # Composite Reward weights: R = alpha * R_Q + beta * sigma(D) + gamma * R_prop
    parser.add_argument('--alpha', type=float, default=0.4, help='Weight for 4-qubit Quantum reward')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for Discriminator realness')
    parser.add_argument('--gamma', type=float, default=0.3, help='Weight for classical property reward')
    parser.add_argument('--prop_metric', type=str, default='qed', choices=['qed', 'sas', 'np', 'dc', 'logp'],
                        help='Target chemical property metric')
    parser.add_argument('--scaffold_bucket_size', type=int, default=25,
                        help='REINVENT Scaffold Diversity Memory bucket size to prevent mode collapse')

    # Checkpoint resumption and steps per epoch
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to existing Actor checkpoint to warm-start or resume training')
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                        help='Number of batch steps per epoch (default: 100 = 1,600 mols/epoch; set 0 for full dataset)')

    # Learning rates
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Learning rate for Actor policy')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for Value Critic')
    parser.add_argument('--lr_disc', type=float, default=1e-4, help='Learning rate for Discriminator')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(" " * 10 + "MolGAN-QRL with Dedicated 1-Qubit = 1-Atom Policy Head")
    print("=" * 75)
    for k, v in vars(args).items():
        print(f"  {k:22s}: {v}")
    print("=" * 75)

    steps_pe = args.steps_per_epoch if args.steps_per_epoch > 0 else None

    solver = SolverQPolicy(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        use_vae_prior=args.use_vae_prior,
        vae_model_path=args.vae_model_path,
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
        resume_checkpoint=args.resume_checkpoint,
        steps_per_epoch=steps_pe,
        device=args.device
    )

    solver.train()


if __name__ == '__main__':
    main()
