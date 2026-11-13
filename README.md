# ⚛️ MolGAN-VQP: 3-Phase Multi-Layer Data Re-Uploading Variational Quantum Policy for De Novo Molecular Generation & Target Screening

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.20+-blueviolet.svg)](https://pennylane.ai/)
[![RDKit](https://img.shields.io/badge/RDKit-2020.09+-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Architectural Grounding:**  
> This framework unifies the foundational theoretical formulation of **Parametrized Quantum Policies for Reinforcement Learning** (*Jerbi et al., NeurIPS 2021*) with the 3-phase hybrid quantum drug discovery pipeline of **MolGAN-QPolicy** (*Naren et al., 2026*).

---

## 📌 3-Phase Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: GRAPH VARIATIONAL AUTOENCODER                    │
│     Molecular Graph (A, X)  ──►  R-GCN + Attention Pooling  ──►  z ~ N(0, I)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ z in R^32 (Continuous Latent Prior)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           PHASE 2: JERBI DATA RE-UPLOADING VARIATIONAL QUANTUM POLICY       │
│                                                                             │
│  • Jerbi Multi-Layer Data Re-Uploading (Degree-L Fourier series expansion)  │
│  • Next Atom Head (9 Qubits = CNS 9-Element Heavy Atom Vocabulary):         │
│      Q0: C  |  Q1: N  |  Q2: O  |  Q3: F  |  Q4: I                          │
│      Q5: P  |  Q6: S  |  Q7: Cl |  Q8: Br                                   │
│  • Next Bond Head (4 Qubits = Kekulé Bond Types, No Aromatic Qubit):       │
│      Q0: No Bond (0) | Q1: Single (1.0) | Q2: Double (2.0) | Q3: Triple (3.0)│
│  • Dynamic Valence Masking (12 Chem Rules) & Scaffold Memory Filter         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ De Novo Generated Molecular Graphs
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             PHASE 3: VERIFICATION MODEL & CNS TARGET SCREENING              │
│                                                                             │
│  • SOTA BBB Ensemble: Deep-CNS-Net + LightGBM + XGBoost + MolGCN            │
│  • Dual BBB & BACE-1 Inhibition Classifier (2058-D Representation)          │
│  • Full Medicinal Chemistry Justification & Actionable 5-Tier Triage        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Upgrades from 4-Qubit QM9 Model to Full 3-Phase CNS Pipeline

The original Variational Quantum Policy implementation in [Yukesh-Naren/MOlGAN-VQP](https://github.com/Yukesh-Naren/MOlGAN-VQP) was restricted to 4 qubits for small QM9 molecules (4 atom types: C, N, O, F; max 9 atoms). In this upgraded system:

1. **Expanded to 9 Qubits for Full CNS Drug Vocabulary**:
   - Explicit one-to-one mapping across 9 real heavy atoms: Carbon (0), Nitrogen (1), Oxygen (2), Fluorine (3), Iodine (4), Phosphorus (5), Sulfur (6), Chlorine (7), Bromine (8).
   - Injects physical bond angles, free valences, and neighborhood chemistries into the data re-uploading input angles.

2. **Kekulé 4-Bond Quantum Policy (No Aromatic Qubit)**:
   - Dedicated 4-qubit Jerbi PQC over legal bond types: None (0), Single (1), Double (2), Triple (3).
   - Dynamic octet rule masking enforces physical valences before Softmax.

3. **Multi-Layer Data Re-Uploading Ansatz (Jerbi et al., NeurIPS 2021)**:
   - At each layer $l \in \{1, \dots, L\}$, continuous molecular subgraph state representations $\mathbf{s}$ are re-encoded:
     $$\phi_{l, i} = \pi \cdot \tanh(\mathbf{w}_{l, i} \cdot \mathbf{s} + b_{l, i})$$
   - Single-qubit Pauli-$Y$ rotations $R_y(\phi_{l, i})$ followed by a circular $\text{CNOT}$ entangling ladder and arbitrary $SU(2)$ Euler rotations $\text{Rot}(\alpha, \beta, \gamma) = R_z(\alpha) R_y(\beta) R_z(\gamma)$.
   - Pauli-$Z$ observable measurement: $\langle \hat{Z}_i \rangle \in [-1, +1]$.
   - Softmax policy scaling: $\pi_\theta(a \mid \mathbf{s}) = \text{Softmax}(\beta \cdot \langle \hat{Z} \rangle)$.

4. **Phase 1 VAE Prior Conditioning**:
   - Generator graph embeddings are directly modulated by Phase 1 Graph VAE latent codes $z \sim \mathcal{N}(0, I)$ in $\mathbb{R}^{32}$.

5. **Dynamic Chemical Valence & Anti-Strain Masking**:
   - Anti-peroxide check ($O-O$ banned).
   - Anti-hydrazine / azo check ($N-N$ banned).
   - Anti-hydroxylamine check ($N-O$ banned).
   - Anti-disulfide check ($S-S$ banned).
   - Monovalent halogen rules (F, Cl, Br, I single-bond to Carbon only).
   - Anti-poly-yne (at most 1 triple bond) and anti-cumulene (no adjacent double bonds).
   - Anti-cage rule (cap backward connections $\le 2$, max rings $\le 2$).
   - Strict ring size allowlist (unstrained 5, 6, 7-membered rings ONLY).
   - Medicinal ring formation incentive (+2.5 single-bond logit boost for unstrained 5/6 rings).

6. **Phase 3 SOTA Screening Integration**:
   - Automated screening through SOTA BBB Ensemble and Dual BACE-1 target classifier.
   - Comprehensive medicinal chemistry justifications and 5-tier triage output.

---

## 📂 Repository Structure

```plaintext
new rotation policy molgan/
├── phase1_autoencoder/           # Phase 1: Graph Variational Autoencoder
│   ├── model.py                  # R-GCN Encoder, Decoder, GraphVAE
│   ├── train_vae.py              # VAE training script
│   ├── sample_vae.py             # Latent prior sampling & interpolation
│   └── vae_model.pt              # Trained VAE checkpoint (z in R^32)
├── molgan_vqp/                   # Phase 2: Jerbi Rotation Policy Generator
│   ├── policy_qc.py              # 9-Qubit Jerbi Multi-Layer Data Re-Uploading PQC
│   ├── circuit.py                # 9-Qubit Quantum PhysChem Reward Circuit
│   ├── models_vqp.py             # ActorRGCN_Jerbi (9 atoms, 4 bonds), Critic, Discriminator
│   ├── ppo_buffer.py             # Rollout buffer for PPO updates
│   ├── reward.py                 # Multi-objective reward engine & Scaffold Memory
│   ├── solver_vqp.py             # End-to-end PPO training engine
│   ├── layers.py                 # Relational Graph Convolutional layers
│   └── main_vqp.py               # CLI entrypoint for training and evaluation
├── phase3_testing_models/        # Phase 3: Verification & Screening Models
│   ├── bace1_dual/               # Multi-Task BBB & BACE-1 target model
│   │   ├── model.py              # DualBBB_BACE1_Classifier (2058-D input)
│   │   ├── features.py           # Morgan FP 2048 + 10 descriptor extraction
│   │   ├── justification.py      # Medicinal chemistry reasoning engine
│   │   ├── predict.py            # Screening API
│   │   ├── bbb_bace1_model.pt    # Pretrained Dual BBB/BACE-1 weights
│   │   └── scaler.pkl            # Pretrained feature scaler
│   └── bbb_sota/                 # SOTA Multi-Representation BBB Ensemble
│       ├── predict_sota_bbb.py   # Ensemble predictor
│       └── master_curated_bbb_dataset.csv
├── data/                         # Datasets & scoring models
│   ├── cns_alzheimers_9_35.smi   # Curated CNS dataset (9 to 35 atoms)
│   ├── SA_score.pkl.gz           # Synthetic Accessibility scoring weights
│   └── NP_score.pkl.gz           # Natural Product-likeness scoring weights
├── checkpoints/                  # Trained Actor and Critic model checkpoints
├── generate_and_screen.py        # End-to-end 3-Phase generation & screening pipeline
├── compare_3_models.py           # Comparative benchmarking across 3 quantum models
├── compare_qm9_models.py         # QM9 benchmarking suite
├── environment.yml               # Complete Conda environment specification
└── README.md                     # Documentation
```

---

## 🚀 Quickstart & Usage

### 1. End-to-End 3-Phase Generation & Live Screening
Run the full 3-phase pipeline connecting Phase 1 (VAE prior), Phase 2 (Jerbi Data Re-Uploading PQC Generator), and Phase 3 (Verification & Screening):

```bash
python generate_and_screen.py \
    --num_molecules 16 \
    --min_atoms 9 \
    --max_atoms 18 \
    --output_dir Results/screened_leads
```

### 2. Training MolGAN-VQP (Phase 2)
Train the 9-qubit Jerbi Variational Quantum Policy generator:

```bash
python molgan_vqp/main_vqp.py \
    --dataset_path data/cns_alzheimers_9_35.sparsedataset \
    --atom_policy jerbi \
    --edge_policy jerbi \
    --n_layers_atom 3 \
    --n_layers_edge 3 \
    --batch_size 16 \
    --epochs 50 \
    --min_atoms 9 \
    --max_atoms 35 \
    --alpha 0.45 \
    --beta 0.10 \
    --gamma 0.30
```

### 3. Sampling from Phase 1 Autoencoder
Sample latent vectors directly from the pre-trained Graph VAE:

```bash
python phase1_autoencoder/sample_vae.py
```

---

## 📜 References
1. Jerbi, S., Gyurik, C., Marshall, S. C., Briegel, H. J., & Dunjko, V. (2021). *Parametrized Quantum Policies for Reinforcement Learning*. Advances in Neural Information Processing Systems (NeurIPS 2021). [arXiv:2103.05577](https://arxiv.org/abs/2103.05577)
2. Naren, Y., Mohithra, K., Maddineni, M., & Dasari, C. M. (2026). *Dedicated Quantum Policy Networks with Scaffold Memory for Diverse De Novo Molecular Generation*.
