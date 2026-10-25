import os
import time
import pickle
import csv
import logging
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Draw

try:
    from .circuit import QuantumPhysChemEvaluator
    from .reward import (
        matrices_to_mols,
        calculate_qed,
        calculate_sas,
        calculate_np,
        calculate_dc,
        calculate_logp,
        CompositeRewardCalculator,
        UNSTABLE_SMARTS_PATTERNS
    )
    from .models_qpolicy import ActorRGCN_Quantum, CriticRGCN, DiscriminatorRGCN
    from .ppo_buffer import PPOBuffer
except (ImportError, ValueError):
    try:
        from phase2_molgan_qpolicy.circuit import QuantumPhysChemEvaluator
        from phase2_molgan_qpolicy.reward import (
            matrices_to_mols,
            calculate_qed,
            calculate_sas,
            calculate_np,
            calculate_dc,
            calculate_logp,
            CompositeRewardCalculator,
            UNSTABLE_SMARTS_PATTERNS
        )
        from phase2_molgan_qpolicy.models_qpolicy import ActorRGCN_Quantum, CriticRGCN, DiscriminatorRGCN
        from phase2_molgan_qpolicy.ppo_buffer import PPOBuffer
    except (ImportError, ValueError):
        from models.circuit import QuantumPhysChemEvaluator
        from models.reward import (
            matrices_to_mols,
            calculate_qed,
            calculate_sas,
            calculate_np,
            calculate_dc,
            calculate_logp,
            CompositeRewardCalculator,
            UNSTABLE_SMARTS_PATTERNS
        )
        from models.models_qpolicy import ActorRGCN_Quantum, CriticRGCN, DiscriminatorRGCN
        from models.ppo_buffer import PPOBuffer


class CNSAlzheimersDataset(Dataset):
    """
    Lightweight PyTorch Dataset loader for CNS Alzheimer's sparse dataset (N <= 35).
    Converts integer node/edge labels into one-hot tensors:
      - Nodes: [N, 5] (PAD:0, C:1, N:2, O:3, F/Halogen:4)
      - Adjacency: [N, N, 5] (None:0, Single:1, Double:2, Triple:3, Aromatic:4)
    """
    def __init__(self, dataset_path):
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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data_X[idx]
        a = self.data_A[idx]
        lbl = self.labels[idx]

        x_t = torch.tensor(x, dtype=torch.long)
        a_t = torch.tensor(a, dtype=torch.long)
        lbl_t = torch.tensor(lbl, dtype=torch.float32)

        x_one_hot = F.one_hot(x_t.clamp(0, 4), num_classes=5).float()
        a_one_hot = F.one_hot(a_t.clamp(0, 4), num_classes=5).float()

        return x_one_hot, a_one_hot, lbl_t


class SolverQPolicy:
    """
    Training Orchestrator for MolGAN-QRL with Dedicated 1-Qubit = 1-Atom Policy Head
    Supports variable-size molecular graph generation across 9 to 35 heavy atoms
    Conditioned directly on Phase 1 Graph VAE latent codes z ~ N(0, I) in R^32.
    """
    def __init__(self,
                 dataset_path='data/cns_alzheimers_9_35.sparsedataset',
                 save_dir='Results/qpolicy_quantum_GAN',
                 batch_size=16,
                 min_atoms=9,
                 max_atoms=35,
                 use_vae_prior=True,
                 vae_model_path=None,
                 epochs=50,
                 ppo_epochs=4,
                 clip_eps=0.2,
                 entropy_coef=0.10,
                 temperature=1.2,
                 alpha=0.4,
                 beta=0.3,
                 gamma=0.3,
                 prop_metric='qed',
                 scaffold_bucket_size=25,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 lr_disc=1e-4,
                 resume_checkpoint=None,
                 steps_per_epoch=50,
                 device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.epochs = epochs
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.steps_per_epoch = steps_per_epoch
        self._data_iter = None
        self.save_dir = save_dir

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

        self.logger.info(f"Initialized Dedicated Q-Policy run at: {self.run_dir}")

        # 1. Dataset & DataLoader
        self.logger.info(f"Loading dataset from {dataset_path}...")
        self.dataset = CNSAlzheimersDataset(dataset_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.dataset_smiles = self.dataset.smiles
        self.max_atoms = self.dataset.vertexes
        self.logger.info(f"Dataset loaded: {len(self.dataset):,} molecules (Max vertices: {self.max_atoms}).")

        # 2. Phase 1 Graph VAE Connection
        self.vae = None
        if use_vae_prior and vae_model_path and os.path.exists(vae_model_path):
            try:
                from phase1_autoencoder.model import GraphVAE
                self.logger.info(f"Loading Phase 1 Graph VAE checkpoint from: {vae_model_path}")
                self.vae = GraphVAE(vertexes=self.max_atoms, edges=5, nodes=10, z_dim=32).to(self.device)
                self.vae.load_state_dict(torch.load(vae_model_path, map_location=self.device))
                self.vae.eval()
                self.logger.info("  ✓ Phase 1 Graph VAE linked directly to Quantum Policy Generator!")
            except Exception as e:
                self.logger.warning(f"Could not load Phase 1 VAE: {e}")

        # 3. Quantum Evaluator & Composite Reward
        self.logger.info("Initializing 4-Qubit Variational Quantum Property Evaluator...")
        self.quantum_evaluator = QuantumPhysChemEvaluator(n_qubits=4, n_layers=3).to(self.device)
        self.reward_calculator = CompositeRewardCalculator(
            quantum_evaluator=self.quantum_evaluator,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            prop_metric=prop_metric,
            scaffold_bucket_size=scaffold_bucket_size
        ).to(self.device)

        # 4. Networks: Actor with Dedicated 1-Qubit = 1-Atom Policy, Critic, Discriminator
        self.logger.info(f"Initializing Actor ({self.min_atoms}-{self.max_atoms} atoms), Critic, and Discriminator...")
        self.actor = ActorRGCN_Quantum(max_atoms=self.max_atoms, num_atom_types=4, num_edge_types=5, z_dim=32).to(self.device)
        self.critic = CriticRGCN(node_dim=5, edge_types=5, z_dim=32).to(self.device)
        self.discriminator = DiscriminatorRGCN(node_dim=5, edge_types=5).to(self.device)

        # Checkpoint Resumption
        self.start_epoch = 1
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.logger.info(f"Loading resume Actor checkpoint from: {resume_checkpoint}")
            self.actor.load_state_dict(torch.load(resume_checkpoint, map_location=self.device))
            import re
            m = re.search(r'actor_epoch_(\d+)\.pt', os.path.basename(resume_checkpoint))
            if m:
                self.start_epoch = int(m.group(1)) + 1
            ckpt_dir = os.path.dirname(resume_checkpoint)
            c_path = os.path.join(ckpt_dir, os.path.basename(resume_checkpoint).replace('actor', 'critic'))
            d_path = os.path.join(ckpt_dir, os.path.basename(resume_checkpoint).replace('actor', 'disc'))
            if os.path.exists(c_path):
                self.critic.load_state_dict(torch.load(c_path, map_location=self.device))
                self.logger.info(f"Loading resume Critic checkpoint from: {c_path}")
            if os.path.exists(d_path):
                self.discriminator.load_state_dict(torch.load(d_path, map_location=self.device))
                self.logger.info(f"Loading resume Discriminator checkpoint from: {d_path}")
            self.logger.info(f"Resuming training starting at Epoch {self.start_epoch} -> {self.epochs}")

        # 5. Rollout Buffer
        self.buffer = PPOBuffer(clip_eps=clip_eps)

        # 6. Optimizers
        self.optimizer_A = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        self.logger.info(f"Dedicated Q-Policy Solver initialized on device: {self.device}")

    def _init_logger(self):
        self.logger = logging.getLogger(f"MolGAN_QPolicy_{time.time()}")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'time_sec', 'loss_d', 'loss_pi', 'loss_v', 'r_avg',
                'valid_rate', 'stable_rate', 'unique_rate', 'novel_rate', 'qed', 'logp', 'sas', 'np', 'dc'
            ])

    def _get_next_batch(self):
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    def train_epoch(self, epoch):
        start_time = time.time()
        self.actor.eval()
        self.critic.eval()
        self.discriminator.train()

        total_loss_d = 0.0
        total_loss_pi = 0.0
        total_loss_v = 0.0
        total_rewards = []
        all_generated_mols = []

        if self.steps_per_epoch is not None:
            num_steps = min(len(self.dataloader), self.steps_per_epoch)
        else:
            num_steps = len(self.dataloader)

        for step in range(num_steps):
            batch_x, batch_adj, batch_labels = self._get_next_batch()
            batch_x = batch_x.to(self.device)
            batch_adj = batch_adj.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Phase 1 VAE Latent Code Encoding (knowledge transfer)
            z_batch = None
            if self.vae is not None:
                with torch.no_grad():
                    # Pad batch_x to 10 atom classes for VAE encoder compatibility if needed
                    B, N, C = batch_x.shape
                    if C < 10:
                        pad_x = torch.zeros(B, N, 10 - C, device=self.device)
                        x_for_vae = torch.cat([batch_x, pad_x], dim=-1)
                    else:
                        x_for_vae = batch_x
                    mu, logvar = self.vae.encode(batch_adj, x_for_vae)
                    z_batch = self.vae.reparameterize(mu, logvar)
            else:
                z_batch = torch.randn(self.batch_size, 32, device=self.device)

            # Phase 2: Environment Rollout with Dedicated 1-Qubit = 1-Atom Policy
            with torch.no_grad():
                nodes, edges, node_feat, adj_t, trajs = self.actor.generate_mols(
                    batch_size=self.batch_size,
                    device=self.device,
                    temperature=self.temperature,
                    min_atoms=self.min_atoms,
                    max_atoms=self.max_atoms,
                    z_cond=z_batch
                )
                old_log_probs, _ = self.actor.evaluate_trajectory_log_probs(trajs, temperature=self.temperature, z_cond=z_batch)
                values = self.critic(node_feat, adj_t, z_cond=z_batch)

                mols = matrices_to_mols(nodes, edges)
                all_generated_mols.extend(mols)

                disc_scores = self.discriminator(node_feat, adj_t)
                rewards, breakdown = self.reward_calculator(mols, disc_scores)

            total_rewards.append(rewards.mean().item())
            self.buffer.store(trajs, old_log_probs, rewards, values)

            # Phase 3: Discriminator Update
            self.optimizer_D.zero_grad()
            dataset_logits = self.discriminator(batch_x, batch_adj).view(-1)
            fake_logits = self.discriminator(node_feat.detach(), adj_t.detach()).view(-1)

            target_dataset = torch.where(batch_labels > 0.5, torch.full_like(dataset_logits, 0.9), torch.zeros_like(dataset_logits))
            loss_d_dataset = F.binary_cross_entropy_with_logits(dataset_logits, target_dataset)
            loss_d_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

            loss_d = loss_d_dataset + loss_d_fake
            loss_d.backward()
            self.optimizer_D.step()
            total_loss_d += loss_d.item()

            # Phase 4: PPO Optimization for Quantum Actor & Classical Critic
            self.actor.train()
            self.critic.train()
            buf_data = self.buffer.get_data()

            for _ in range(self.ppo_epochs):
                self.optimizer_A.zero_grad()
                new_log_probs, entropy = self.actor.evaluate_trajectory_log_probs(buf_data['trajectories'], temperature=self.temperature, z_cond=z_batch)
                log_ratio = torch.clamp(new_log_probs - buf_data['old_log_probs'], -5.0, 5.0)
                ratio = torch.exp(log_ratio)
                advantages = buf_data['advantages']

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                loss_pi = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.optimizer_A.step()

                self.optimizer_C.zero_grad()
                curr_values = self.critic(node_feat.detach(), adj_t.detach(), z_cond=z_batch)
                loss_v = F.mse_loss(curr_values, buf_data['returns'])

                loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.optimizer_C.step()

                total_loss_pi += loss_pi.item()
                total_loss_v += loss_v.item()

            self.buffer.clear()

            if (step + 1) % 10 == 0 or (step + 1) == num_steps:
                val_cnt = sum(1 for m in mols if m is not None)
                self.logger.info(
                    f"  Epoch [{epoch}/{self.epochs}] Step [{step + 1}/{num_steps}] | "
                    f"Batch R_avg: {rewards.mean().item():.3f} | L_D: {loss_d.item():.3f} | "
                    f"Val: {val_cnt}/{len(mols)}"
                )

        elapsed = time.time() - start_time
        metrics = self._evaluate_and_log(epoch, elapsed, total_loss_d / num_steps,
                                         total_loss_pi / (num_steps * self.ppo_epochs),
                                         total_loss_v / (num_steps * self.ppo_epochs),
                                         np.mean(total_rewards), all_generated_mols)
        return metrics

    def _evaluate_and_log(self, epoch, elapsed, loss_d, loss_pi, loss_v, r_avg, mols):
        valid_mols = [m for m in mols if m is not None]
        valid_rate = len(valid_mols) / float(len(mols)) if mols else 0.0

        smiles_list = [Chem.MolToSmiles(m) for m in valid_mols if Chem.MolToSmiles(m)]
        unique_rate = len(set(smiles_list)) / float(len(smiles_list)) if smiles_list else 0.0
        novel_rate = sum(1 for s in smiles_list if s not in self.dataset_smiles) / float(len(smiles_list)) if smiles_list else 0.0

        stable_count = 0
        for m in valid_mols:
            has_bad = False
            for pat in UNSTABLE_SMARTS_PATTERNS:
                if pat is not None and m.HasSubstructMatch(pat):
                    has_bad = True
                    break
            if not has_bad:
                stable_count += 1
        stable_rate = stable_count / float(len(valid_mols)) if valid_mols else 0.0

        qeds = [calculate_qed(m) for m in valid_mols]
        logps = [calculate_logp(m, norm=False) for m in valid_mols]
        sass = [calculate_sas(m, norm=False) for m in valid_mols]
        nps = [calculate_np(m, norm=False) for m in valid_mols]
        dcs = [calculate_dc(m) for m in valid_mols]

        m_qed = np.mean(qeds) if qeds else 0.0
        m_logp = np.mean(logps) if logps else 0.0
        m_sas = np.mean(sass) if sass else 0.0
        m_np = np.mean(nps) if nps else 0.0
        m_dc = np.mean(dcs) if dcs else 0.0

        self.logger.info(
            f"Epoch [{epoch}/{self.epochs}] ({elapsed:.1f}s) | "
            f"L_D: {loss_d:.3f} | L_pi: {loss_pi:.3f} | L_v: {loss_v:.3f} | R_avg: {r_avg:.3f} | "
            f"Val: {valid_rate*100:.1f}% | Stable: {stable_rate*100:.1f}% | Uniq: {unique_rate*100:.1f}% | Nov: {novel_rate*100:.1f}% | "
            f"QED: {m_qed:.2f} | LogP: {m_logp:.2f} | SAS: {m_sas:.2f} | NP: {m_np:.2f}"
        )

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, round(elapsed, 2), round(loss_d, 4), round(loss_pi, 4), round(loss_v, 4), round(r_avg, 4),
                round(valid_rate, 4), round(stable_rate, 4), round(unique_rate, 4), round(novel_rate, 4),
                round(m_qed, 4), round(m_logp, 4), round(m_sas, 4), round(m_np, 4), round(m_dc, 4)
            ])

        # Checkpoint saving every epoch
        ckpt_a = os.path.join(self.model_dir, f'actor_epoch_{epoch}.pt')
        ckpt_c = os.path.join(self.model_dir, f'critic_epoch_{epoch}.pt')
        ckpt_d = os.path.join(self.model_dir, f'disc_epoch_{epoch}.pt')
        torch.save(self.actor.state_dict(), ckpt_a)
        torch.save(self.critic.state_dict(), ckpt_c)
        torch.save(self.discriminator.state_dict(), ckpt_d)

        # Mirror latest checkpoint to checkpoints/ for immediate generation CLI access
        static_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
        if os.path.exists(static_ckpt):
            torch.save(self.actor.state_dict(), os.path.join(static_ckpt, f'actor_epoch_{epoch}.pt'))
            torch.save(self.critic.state_dict(), os.path.join(static_ckpt, f'critic_epoch_{epoch}.pt'))
            torch.save(self.discriminator.state_dict(), os.path.join(static_ckpt, f'disc_epoch_{epoch}.pt'))

        # Render sample images
        if valid_mols:
            img_path = os.path.join(self.img_dir, f'epoch_{epoch}.png')
            legends = [
                f"QED:{calculate_qed(m):.2f}|LogP:{calculate_logp(m, norm=False):.1f}|Rings:{m.GetRingInfo().NumRings()}"
                for m in valid_mols[:16]
            ]
            img = Draw.MolsToGridImage(valid_mols[:16], molsPerRow=4, subImgSize=(250, 250), legends=legends)
            img.save(img_path)

        return {'valid_rate': valid_rate, 'unique_rate': unique_rate, 'novel_rate': novel_rate}

    def train(self):
        self.logger.info("=" * 75)
        self.logger.info(f"Starting Training: Epochs {self.start_epoch} -> {self.epochs} | Batch: {self.batch_size} | Device: {self.device}")
        self.logger.info("=" * 75)
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_epoch(epoch)
        self.logger.info("Training completed successfully!")
