import os
import sys
import time
import csv
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Draw

from molgan_vqp.circuit import QuantumPhysChemEvaluator
from molgan_vqp.reward import (
    matrices_to_mols,
    calculate_qed,
    calculate_sas,
    calculate_np,
    calculate_dc,
    calculate_logp,
    CompositeRewardCalculator,
    UNSTABLE_SMARTS_PATTERNS
)
from molgan_vqp.models_vqp import ActorRGCN_Jerbi, ActorRGCN_Quantum, CriticRGCN, DiscriminatorRGCN
from molgan_vqp.ppo_buffer import PPOBuffer


class MolecularDataset(Dataset):
    """
    PyTorch Dataset loader for molecular sparse datasets (CNS Alzheimer's or QM9).
    Converts integer node/edge labels into one-hot tensors:
      - Nodes: [N, node_dim] (PAD:0, C:1, N:2, O:3, F:4, I:5, P:6, S:7, Cl:8, Br:9)
      - Adjacency: [N, N, 4] (None:0, Single:1, Double:2, Triple:3)
    """
    def __init__(self, dataset_path, node_dim=10, edge_types=4):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        self.data_A = data['data_A']  # [M, N, N] integer indices
        self.data_X = data['data_X']  # [M, N] integer indices
        self.labels = data.get('labels', None)
        if self.labels is None:
            self.labels = np.ones(len(self.data_X), dtype=np.float32)

        self.smiles = set(data['smiles']) if 'smiles' in data else set()
        self.num_samples = len(self.data_X)
        self.vertexes = self.data_X.shape[1]
        self.node_dim = node_dim
        self.edge_types = edge_types

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data_X[idx]
        a = self.data_A[idx]
        lbl = self.labels[idx]

        x_t = torch.tensor(x, dtype=torch.long)
        a_t = torch.tensor(a, dtype=torch.long)
        lbl_t = torch.tensor(lbl, dtype=torch.float32)

        x_one_hot = F.one_hot(x_t.clamp(0, self.node_dim - 1), num_classes=self.node_dim).float()
        # Clamp to 4 bond types (0: None, 1: Single, 2: Double, 3: Triple)
        a_one_hot = F.one_hot(a_t.clamp(0, self.edge_types - 1), num_classes=self.edge_types).float()

        return x_one_hot, a_one_hot, lbl_t


QM9Dataset = MolecularDataset
CNSAlzheimersDataset = MolecularDataset


class SolverJerbiVQP:
    """
    Training Solver for MolGAN-VQP with Jerbi et al. (NeurIPS 2021)
    Variational Quantum Policies (Softmax-PQC with Multi-Layer Data Re-Uploading Rotation Policy).
    Supports variable-size molecular graph generation across 9 to 35 heavy atoms,
    with 9-qubit element-wise policy, 4-bond edge policy without aromatic bond,
    and conditioning on Phase 1 Graph VAE latent codes z ~ N(0, I) in R^32.
    """
    def __init__(self,
                 dataset_path='data/cns_alzheimers_9_35.sparsedataset',
                 save_dir='Results/vqp_quantum_GAN',
                 batch_size=16,
                 min_atoms=9,
                 max_atoms=35,
                 use_vae_prior=True,
                 vae_model_path=None,
                 steps_per_epoch=100,
                 resume_checkpoint=None,
                 epochs=50,
                 ppo_epochs=4,
                 clip_eps=0.2,
                 entropy_coef=0.10,
                 temperature=1.2,
                 alpha=0.45,
                 beta=0.10,
                 gamma=0.30,
                 prop_metric='bace1',
                 scaffold_bucket_size=25,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 lr_disc=1e-4,
                 n_layers_atom=3,
                 n_layers_edge=3,
                 output_scale=3.0,
                 atom_policy_type='jerbi',
                 edge_policy_type='jerbi',
                 use_valence_mask=True,
                 use_filter=True,
                 use_scaffold_memory=True,
                 use_tanimoto=True,
                 enforce_connectivity=True,
                 use_atom_filters=True,
                 device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.atom_policy_type = atom_policy_type
        self.edge_policy_type = edge_policy_type
        self.n_layers_atom = n_layers_atom
        self.n_layers_edge = n_layers_edge
        self.output_scale = output_scale
        self.use_valence_mask = use_valence_mask
        self.use_filter = use_filter
        self.use_scaffold_memory = use_scaffold_memory
        self.use_tanimoto = use_tanimoto
        self.enforce_connectivity = enforce_connectivity
        self.use_atom_filters = use_atom_filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.ppo_epochs = ppo_epochs
        self.steps_per_epoch = steps_per_epoch
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.save_dir = save_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prop_metric = prop_metric
        self.scaffold_bucket_size = scaffold_bucket_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_disc = lr_disc
        self._data_iter = None

        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = os.path.join(save_dir, timestamp)
        self.log_dir = os.path.join(self.run_dir, 'log_dir')
        self.model_dir = os.path.join(self.run_dir, 'model_dir')
        self.img_dir = os.path.join(self.run_dir, 'img_dir')

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, 'train.log')
        self.csv_file = os.path.join(self.log_dir, 'metrics.csv')
        self._init_logger()
        self._init_csv()

        self.logger.info(f"Initialized Jerbi VQP Policy run at: {self.run_dir}")

        # 1. Dataset & DataLoader (CNS 9-35 or QM9)
        self.logger.info(f"Loading dataset from {dataset_path}...")
        self.dataset = MolecularDataset(dataset_path, node_dim=10, edge_types=4)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.dataset_smiles = self.dataset.smiles
        self.max_atoms = max(self.max_atoms, self.dataset.vertexes)
        self.logger.info(f"Dataset loaded: {len(self.dataset):,} molecules (Max vertices: {self.dataset.vertexes}, Effective max_atoms: {self.max_atoms}).")

        # 2. Phase 1 Graph VAE Connection
        self.vae = None
        if use_vae_prior and vae_model_path and os.path.exists(vae_model_path):
            try:
                from phase1_autoencoder.model import GraphVAE
                self.logger.info(f"Loading Phase 1 Graph VAE checkpoint from: {vae_model_path}")
                self.vae = GraphVAE(vertexes=self.max_atoms, edges=4, nodes=10, z_dim=32).to(self.device)
                self.vae.load_state_dict(torch.load(vae_model_path, map_location=self.device), strict=False)
                self.vae.eval()
                self.logger.info("  ✓ Phase 1 Graph VAE linked directly to Jerbi Quantum Policy Generator!")
            except Exception as e:
                self.logger.warning(f"Notice: Phase 1 VAE initialization exception ({e}), using standard normal prior.")

        # 3. Quantum Evaluator & Composite Reward
        self.logger.info("Initializing 9-Qubit Variational Quantum Property Evaluator...")
        self.quantum_evaluator = QuantumPhysChemEvaluator(n_qubits=9, n_layers=3).to(self.device)
        self.reward_calculator = CompositeRewardCalculator(
            quantum_evaluator=self.quantum_evaluator,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            prop_metric=prop_metric,
            scaffold_bucket_size=scaffold_bucket_size,
            use_filter=self.use_filter,
            use_scaffold_memory=self.use_scaffold_memory,
            use_tanimoto=self.use_tanimoto
        ).to(self.device)

        # 4. Networks: Actor with Jerbi Rotation Policy Head, Critic, Discriminator
        self.logger.info(f"Initializing Jerbi Actor ({self.min_atoms}-{self.max_atoms} atoms, Atom: {self.atom_policy_type}, Edge: {self.edge_policy_type}, Masking: {self.use_valence_mask}), Critic, Discriminator...")
        self.actor = ActorRGCN_Jerbi(
            max_atoms=self.max_atoms,
            min_atoms=self.min_atoms,
            num_atom_types=9,
            num_edge_types=4,
            node_dim=10,
            z_dim=32,
            n_layers_atom=self.n_layers_atom,
            n_layers_edge=self.n_layers_edge,
            output_scale=self.output_scale,
            atom_policy_type=self.atom_policy_type,
            edge_policy_type=self.edge_policy_type,
            use_valence_mask=self.use_valence_mask,
            enforce_connectivity=self.enforce_connectivity,
            use_atom_filters=self.use_atom_filters
        ).to(self.device)
        self.critic = CriticRGCN(node_dim=10, edge_types=4, z_dim=32).to(self.device)
        self.discriminator = DiscriminatorRGCN(node_dim=10, edge_types=4).to(self.device)

        # Checkpoint Resumption
        self.start_epoch = 1
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.logger.info(f"Loading resume Actor checkpoint from: {resume_checkpoint}")
            state_dict = torch.load(resume_checkpoint, map_location=self.device)
            model_dict = self.actor.state_dict()
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
            self.actor.load_state_dict(filtered_dict, strict=False)

            critic_ckpt = resume_checkpoint.replace('actor_', 'critic_')
            disc_ckpt = resume_checkpoint.replace('actor_', 'disc_')
            if os.path.exists(critic_ckpt):
                self.logger.info(f"Loading resume Critic checkpoint from: {critic_ckpt}")
                c_state = torch.load(critic_ckpt, map_location=self.device)
                c_model = self.critic.state_dict()
                c_filtered = {k: v for k, v in c_state.items() if k in c_model and v.shape == c_model[k].shape}
                self.critic.load_state_dict(c_filtered, strict=False)
            if os.path.exists(disc_ckpt):
                self.logger.info(f"Loading resume Discriminator checkpoint from: {disc_ckpt}")
                d_state = torch.load(disc_ckpt, map_location=self.device)
                d_model = self.discriminator.state_dict()
                d_filtered = {k: v for k, v in d_state.items() if k in d_model and v.shape == d_model[k].shape}
                self.discriminator.load_state_dict(d_filtered, strict=False)

        # 5. Rollout Buffer
        self.buffer = PPOBuffer(clip_eps=clip_eps)

        # 6. Optimizers
        self.optimizer_A = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        self.best_reward = -float('inf')
        self.logger.info(f"Jerbi VQP Solver initialized on device: {self.device}")

    def _init_logger(self):
        self.logger = logging.getLogger(f"MolGAN_VQP_{time.time()}")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'loss_actor', 'loss_critic', 'loss_disc',
                'mean_reward', 'mean_val', 'mean_bace1', 'mean_bbb', 'mean_sas',
                'validity', 'uniqueness', 'novelty', 'diversity', 'stability',
                'time_sec'
            ])

    def get_real_batch(self):
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            batch = next(self._data_iter)
        return batch[0].to(self.device), batch[1].to(self.device)

    def train(self):
        self.logger.info(f"Starting Jerbi VQP PPO Training for {self.epochs} epochs...")
        total_steps_per_epoch = self.steps_per_epoch if self.steps_per_epoch else len(self.dataloader)

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            self.actor.train()
            self.critic.train()
            self.discriminator.train()

            epoch_rewards = []
            epoch_loss_a = []
            epoch_loss_c = []
            epoch_loss_d = []
            epoch_bace1 = []
            epoch_bbb = []
            epoch_sas = []
            epoch_stability = []

            all_gen_smiles = []

            for step in range(total_steps_per_epoch):
                # 1. Sample Phase 1 Graph VAE latent prior z ~ N(0, I) in R^32
                z_cond = torch.randn(self.batch_size, 32, device=self.device)

                # 2. Rollout Generation via Jerbi Rotation Policy
                node_idx, adj_idx, node_feat, adj_t, trajs = self.actor.generate_mols(
                    batch_size=self.batch_size,
                    device=self.device,
                    temperature=self.temperature,
                    min_atoms=self.min_atoms,
                    max_atoms=self.max_atoms,
                    z_cond=z_cond
                )

                # 3. Convert generated tensors to RDKit molecules
                mols = matrices_to_mols(node_idx, adj_idx)
                for m in mols:
                    if m is not None:
                        smi = Chem.MolToSmiles(m)
                        if smi:
                            all_gen_smiles.append(smi)

                # 4. Evaluate Discriminator realness and Composite Rewards
                with torch.no_grad():
                    active_mask = (node_idx > 0).float()
                    disc_scores = self.discriminator(node_feat, adj_t, node_mask=active_mask)
                    rewards, breakdown = self.reward_calculator(mols, discriminator_scores=disc_scores)
                    values = self.critic(node_feat, adj_t, node_mask=active_mask, z_cond=z_cond)
                    old_log_probs, _ = self.actor.evaluate_trajectory_log_probs(trajs, temperature=self.temperature, z_cond=z_cond)

                # 5. Store in PPO Rollout Buffer
                self.buffer.store(trajs, old_log_probs, rewards, values)

                # 6. PPO Update Phase
                buf_data = self.buffer.get_data()
                advantages = buf_data['advantages'].detach()
                returns = buf_data['returns'].detach()
                old_lp = buf_data['old_log_probs'].detach()
                stored_trajs = buf_data['trajectories']

                for _ in range(self.ppo_epochs):
                    curr_lp, entropy = self.actor.evaluate_trajectory_log_probs(stored_trajs, temperature=self.temperature, z_cond=z_cond)
                    ratio = torch.exp(curr_lp - old_lp)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                    loss_policy = -torch.min(surr1, surr2).mean()

                    curr_val = self.critic(node_feat, adj_t, node_mask=active_mask, z_cond=z_cond)
                    loss_val = F.mse_loss(curr_val, returns)

                    loss_actor = loss_policy - self.entropy_coef * entropy

                    self.optimizer_A.zero_grad()
                    loss_actor.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.optimizer_A.step()

                    self.optimizer_C.zero_grad()
                    loss_val.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.optimizer_C.step()

                # 7. Discriminator GAN Update
                real_nodes, real_adj = self.get_real_batch()
                real_mask = (real_nodes.argmax(dim=-1) > 0).float()
                d_real = self.discriminator(real_nodes, real_adj, node_mask=real_mask)
                d_fake = self.discriminator(node_feat.detach(), adj_t.detach(), node_mask=active_mask)

                loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)

                self.optimizer_D.zero_grad()
                loss_d.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.optimizer_D.step()

                self.buffer.clear()

                epoch_rewards.append(rewards.mean().item())
                epoch_loss_a.append(loss_actor.item())
                epoch_loss_c.append(loss_val.item())
                epoch_loss_d.append(loss_d.item())
                epoch_bace1.append(breakdown['p_bace1'].mean().item())
                epoch_bbb.append(breakdown['p_bbb'].mean().item())
                epoch_sas.append(breakdown['r_sas'].mean().item())
                epoch_stability.append(breakdown['stability_rate'])

            elapsed = time.time() - t0

            # Compute generation metrics
            total_mols = len(all_gen_smiles)
            unique_mols = len(set(all_gen_smiles))
            novel_mols = sum(1 for s in set(all_gen_smiles) if s not in self.dataset_smiles)

            validity = total_mols / max(total_steps_per_epoch * self.batch_size, 1)
            uniqueness = unique_mols / max(total_mols, 1)
            novelty = novel_mols / max(unique_mols, 1)
            diversity = uniqueness  # Approximate diversity proxy
            stability = np.mean(epoch_stability) if epoch_stability else 0.0

            m_rew = np.mean(epoch_rewards)
            m_la = np.mean(epoch_loss_a)
            m_lc = np.mean(epoch_loss_c)
            m_ld = np.mean(epoch_loss_d)
            m_bace = np.mean(epoch_bace1)
            m_bbb = np.mean(epoch_bbb)
            m_sas = np.mean(epoch_sas)

            self.logger.info(
                f"Epoch [{epoch:03d}/{self.epochs:03d}] ({elapsed:.1f}s) | "
                f"Rew: {m_rew:+.3f} | BACE-1: {m_bace:.3f} | BBB: {m_bbb:.3f} | "
                f"Val: {validity*100:.1f}% | Uniq: {uniqueness*100:.1f}% | Nov: {novelty*100:.1f}% | "
                f"Stab: {stability*100:.1f}% | L_A: {m_la:.3f} | L_C: {m_lc:.3f} | L_D: {m_ld:.3f}"
            )

            # Write CSV metrics
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, m_la, m_lc, m_ld, m_rew, 0.0, m_bace, m_bbb, m_sas,
                    validity, uniqueness, novelty, diversity, stability, elapsed
                ])

            # Save Checkpoints
            actor_path = os.path.join(self.model_dir, f"actor_epoch_{epoch}.pt")
            critic_path = os.path.join(self.model_dir, f"critic_epoch_{epoch}.pt")
            disc_path = os.path.join(self.model_dir, f"disc_epoch_{epoch}.pt")
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)
            torch.save(self.discriminator.state_dict(), disc_path)

            # Always save latest checkpoint
            torch.save(self.actor.state_dict(), os.path.join(self.model_dir, "actor_latest.pt"))

            # Save best checkpoint
            if m_rew > self.best_reward:
                self.best_reward = m_rew
                torch.save(self.actor.state_dict(), os.path.join(self.model_dir, "actor_best.pt"))
                torch.save(self.critic.state_dict(), os.path.join(self.model_dir, "critic_best.pt"))
                torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, "disc_best.pt"))
                self.logger.info(f"  ★ New best model achieved at epoch {epoch} with Reward = {m_rew:+.4f} (Saved to actor_best.pt)")

            # Save Molecular Grid Image
            if mols:
                valid_mols = [m for m in mols if m is not None]
                if valid_mols:
                    img_path = os.path.join(self.img_dir, f"epoch_{epoch:03d}_mols.png")
                    try:
                        img = Draw.MolsToGridImage(valid_mols[:16], molsPerRow=4, subImgSize=(250, 250))
                        img.save(img_path)
                    except Exception:
                        pass

        self.logger.info("✓ Jerbi VQP PPO Training Complete!")


# Alias SolverQPolicy for seamless compatibility with final phase 2
SolverQPolicy = SolverJerbiVQP
