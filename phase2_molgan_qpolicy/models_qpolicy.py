import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from .layers import RelationalGCN
    from .quantum_policy import DedicatedAtomQuantumPolicy
except (ImportError, ValueError):
    try:
        from phase2_molgan_qpolicy.layers import RelationalGCN
        from phase2_molgan_qpolicy.quantum_policy import DedicatedAtomQuantumPolicy
    except (ImportError, ValueError):
        from models.layers import RelationalGCN
        from models.quantum_policy import DedicatedAtomQuantumPolicy

# Atomic limits and valence rules
MAX_VALENCES = {
    0: 0,  # PAD
    1: 4,  # C
    2: 3,  # N
    3: 2,  # O
    4: 1   # F (and halogens)
}

# Bond orders matching bond type index:
# 0: None, 1: Single (1.0), 2: Double (2.0), 3: Triple (3.0), 4: Aromatic (1.5)
BOND_ORDERS = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 1.5
}


class ActorRGCN_Quantum(nn.Module):
    """
    Autoregressive Graph Convolutional Actor with Dedicated 1-Qubit = 1-Atom Policy Head
    Supports variable-size molecular graph generation across the CNS target range: 9 to 35 heavy atoms.
    Directly conditioned by Phase 1 Graph VAE latent codes z ~ N(0, I) in R^32.
    """
    def __init__(self, max_atoms=35, num_atom_types=4, num_edge_types=5,
                 node_dim=5, hidden_dims=[128, 64], dropout_rate=0.0, z_dim=32):
        super(ActorRGCN_Quantum, self).__init__()
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.num_edge_types = num_edge_types
        self.node_dim = node_dim
        self.z_dim = z_dim

        # Relational GCN encoder
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=num_edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        # VAE Latent Adapter: connects Phase 1 Graph VAE latent representation z directly to generator
        self.z_adapter = nn.Sequential(
            nn.Linear(z_dim, self.encoder.out_dim),
            nn.LeakyReLU(0.2)
        )

        # Dedicated 1-Qubit = 1-Atom Quantum Policy Head (Q-Actor)
        self.quantum_policy = DedicatedAtomQuantumPolicy(
            in_features=self.encoder.out_dim,
            n_qubits=4,
            n_layers=3
        )

        # Edge formation head: Takes pair embeddings [h_u, h_v, h_G] -> 5 bond types
        pair_in_dim = self.encoder.out_dim * 3
        self.edge_head = nn.Sequential(
            nn.Linear(pair_in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_edge_types)
        )

    def forward_subgraph(self, node_features, adj_tensor, active_mask, z_cond=None):
        """Encodes active subgraph and optionally modulates with Phase 1 VAE latent vector."""
        node_emb, graph_emb = self.encoder(node_features, adj_tensor, node_mask=active_mask)
        if z_cond is not None:
            z_proj = self.z_adapter(z_cond)
            graph_emb = graph_emb + z_proj
        return node_emb, graph_emb

    def compute_valence_mask(self, node_indices, adj_indices, u, v):
        """
        Dynamically calculates legal bond types between atom u and atom v
        based on quantum physical valences and chemical stability rules.
        """
        batch_size = node_indices.size(0)
        device = node_indices.device
        mask = torch.ones(batch_size, self.num_edge_types, dtype=torch.bool, device=device)

        for b in range(batch_size):
            type_u = int(node_indices[b, u].item())
            type_v = int(node_indices[b, v].item())

            # If either atom is PAD (0), only 'No bond' (0) is legal
            if type_u == 0 or type_v == 0:
                mask[b, 1:] = False
                continue

            # Anti-peroxide check: Strict ban on O-O bonds
            if type_u == 3 and type_v == 3:
                mask[b, 1:] = False
                continue

            # Anti-hetero-fluorine check: F can only bond to Carbon (type 1)
            if (type_u == 4 and type_v != 1) or (type_v == 4 and type_u != 1):
                mask[b, 1:] = False
                continue

            # Anti-hydrazine / azo check: Strict ban on N-N bonds (all nitrogens must anchor to Carbon)
            if type_u == 2 and type_v == 2:
                mask[b, 1:] = False
                continue

            # Anti-hydroxylamine check: Strict ban on N-O bonds (heteroatoms must anchor to Carbon)
            if (type_u == 2 and type_v == 3) or (type_u == 3 and type_v == 2):
                mask[b, 1:] = False
                continue

            # Calculate current utilized valence
            curr_val_u = sum(BOND_ORDERS[adj_indices[b, u, w].item()] for w in range(u) if w != v)
            curr_val_v = sum(BOND_ORDERS[adj_indices[b, v, w].item()] for w in range(u) if w != u)

            rem_val_u = MAX_VALENCES[type_u] - curr_val_u
            rem_val_v = MAX_VALENCES[type_v] - curr_val_v

            # Check each bond order
            for bt in range(1, self.num_edge_types):
                req_order = BOND_ORDERS[bt]
                if req_order > rem_val_u or req_order > rem_val_v:
                    mask[b, bt] = False

            # Anti-poly-yne rule: at most 1 triple bond per molecule
            num_triple = sum(1 for w1 in range(u) for w2 in range(w1 + 1, u) if adj_indices[b, w1, w2].item() == 3)
            if num_triple >= 1:
                mask[b, 3] = False  # Cap triple bonds to prevent poly-yne chains

            # Anti-cumulene / allene rule: prevent multiple double bonds on the same atom
            u_has_double = any(adj_indices[b, u, w].item() == 2 for w in range(u) if w != v)
            v_has_double = any(adj_indices[b, v, w].item() == 2 for w in range(u) if w != u)
            if u_has_double or v_has_double:
                mask[b, 2] = False  # Forces alternating single/double bonds (enables aromatic ring systems)

            # Anti-strained 3-membered ring rule for multiple bonds:
            shares_neighbor = any(adj_indices[b, u, w].item() > 0 and adj_indices[b, v, w].item() > 0 for w in range(u) if w != v)
            if shares_neighbor:
                mask[b, 2:] = False

        return mask

    def generate_mols(self, batch_size, device='cpu', temperature=1.2,
                      min_atoms=9, max_atoms=None, target_sizes=None, z_cond=None):
        """
        Autoregressively generates molecules with flexible size between min_atoms and max_atoms (9 to 35).
        Conditioned directly on Phase 1 Graph VAE latent codes z_cond.
        """
        actual_max = max_atoms if max_atoms is not None else self.max_atoms

        # Sample or enforce target sizes in [min_atoms, actual_max]
        if target_sizes is None:
            target_sizes = torch.randint(min_atoms, actual_max + 1, (batch_size,), device=device)
        elif isinstance(target_sizes, int):
            target_sizes = torch.full((batch_size,), target_sizes, dtype=torch.long, device=device)
        else:
            target_sizes = target_sizes.to(device)

        # Default VAE prior if not supplied
        if z_cond is None:
            z_cond = torch.randn(batch_size, self.z_dim, device=device)
        else:
            z_cond = z_cond.to(device)

        node_features = torch.zeros(batch_size, actual_max, self.node_dim, device=device)
        node_features[:, :, 0] = 1.0  # Default to PAD
        node_indices = torch.zeros((batch_size, actual_max), dtype=torch.long, device=device)

        adj_tensor = torch.zeros(batch_size, actual_max, actual_max, self.num_edge_types, device=device)
        adj_tensor[:, :, :, 0] = 1.0  # Default to No bond
        adj_indices = torch.zeros(batch_size, actual_max, actual_max, dtype=torch.long, device=device)

        active_mask = torch.zeros(batch_size, actual_max, device=device)
        trajectories = []

        # Autoregressive growth loop
        for step in range(actual_max):
            # 1. Encode intermediate graph with VAE latent conditioning
            node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask, z_cond=z_cond)

            # 2. Extract local geometric parameters for Quantum Policy Head
            if step == 0:
                bond_angles = torch.full((batch_size,), 1.91, device=device)  # 109.5 deg
                free_val = torch.full((batch_size,), 4.0, device=device)
                neighbor_types = torch.ones(batch_size, dtype=torch.long, device=device)  # Carbon anchor
            else:
                neighbor_types = node_indices[:, step - 1]
                free_val_list = []
                for b in range(batch_size):
                    anchor_t = int(neighbor_types[b].item())
                    if anchor_t == 0:
                        free_val_list.append(0.0)
                    else:
                        used = sum(BOND_ORDERS[adj_indices[b, step - 1, w].item()] for w in range(step))
                        free_val_list.append(max(0.0, MAX_VALENCES[anchor_t] - used))
                free_val = torch.tensor(free_val_list, device=device)
                bond_angles = torch.full((batch_size,), 1.91, device=device)

            # 3. Quantum Policy Head execution
            q_logits, q_probs = self.quantum_policy(graph_emb, bond_angles, free_val, neighbor_types)

            # Enforce Carbon anchor on Step 0 and dynamically scale heteroatoms according to medicinal chemistry
            if step == 0:
                q_logits[:, 1:] = -1e9
            else:
                for b in range(batch_size):
                    tgt_sz = target_sizes[b].item()

                    # 1. Total heteroatom cap: enforce Carbon >= 75%
                    max_hetero = max(2, int(0.25 * tgt_sz))
                    num_hetero = (node_indices[b, :step] > 1).sum().item()
                    if num_hetero >= max_hetero:
                        q_logits[b, 1:] = -1e9  # Remaining atoms must be Carbon
                        continue

                    # 2. Strict Nitrogen cap: at most 12-14% of atoms (typically 1-3 nitrogens)
                    max_n = max(1, int(0.14 * tgt_sz))
                    num_n = (node_indices[b, :step] == 2).sum().item()
                    if num_n >= max_n:
                        q_logits[b, 1] = -1e9  # Nitrogen (action 1) masked out

                    # 3. Oxygen cap: at most 15% of atoms
                    max_o = max(1, int(0.15 * tgt_sz))
                    num_o = (node_indices[b, :step] == 3).sum().item()
                    if num_o >= max_o:
                        q_logits[b, 2] = -1e9  # Oxygen (action 2) masked out

                    # 4. Fluorine cap: at most 2 fluorines
                    num_f = (node_indices[b, :step] == 4).sum().item()
                    if num_f >= 2:
                        q_logits[b, 3] = -1e9  # Fluorine (action 3) masked out

            node_mask = (q_logits > -1e8)
            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            node_action = node_dist.sample()         # in {0, 1, 2, 3}
            node_log_prob = node_dist.log_prob(node_action)

            # Map action {0, 1, 2, 3} to atom type {1: C, 2: N, 3: O, 4: F}
            atom_types = node_action + 1

            step_record = {
                'step': step,
                'state_node': node_features.clone(),
                'state_adj': adj_tensor.clone(),
                'active_mask': active_mask.clone(),
                'bond_angles': bond_angles.clone(),
                'free_val': free_val.clone(),
                'neighbor_types': neighbor_types.clone(),
                'node_mask': node_mask.clone(),
                'node_action': node_action,
                'node_log_prob': node_log_prob,
                'edge_records': []
            }

            for b in range(batch_size):
                if step < target_sizes[b]:
                    a_type = atom_types[b].item()
                    node_indices[b, step] = a_type
                    node_features[b, step, :] = 0.0
                    node_features[b, step, a_type] = 1.0
                    active_mask[b, step] = 1.0
                else:
                    # PAD for completed molecules
                    node_indices[b, step] = 0
                    node_features[b, step, :] = 0.0
                    node_features[b, step, 0] = 1.0
                    active_mask[b, step] = 0.0

            # 4. Edge formation head
            if step > 0:
                node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask, z_cond=z_cond)
                u = step

                for v in range(step):
                    h_u = node_emb[:, u, :]
                    h_v = node_emb[:, v, :]
                    pair_feat = torch.cat([h_u, h_v, graph_emb], dim=-1)

                    edge_logits = self.edge_head(pair_feat)

                    # Valence and Aromatic Masking
                    valence_mask = self.compute_valence_mask(node_indices, adj_indices, u, v)
                    edge_logits = torch.where(valence_mask, edge_logits, torch.tensor(-1e9, device=device))

                    edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                    edge_action = edge_dist.sample()
                    edge_log_prob = edge_dist.log_prob(edge_action)

                    step_record['edge_records'].append({
                        'u': u,
                        'v': v,
                        'edge_action': edge_action,
                        'edge_log_prob': edge_log_prob,
                        'valence_mask': valence_mask.clone()
                    })

                    for b in range(batch_size):
                        if step < target_sizes[b] and v < target_sizes[b]:
                            b_type = edge_action[b].item()
                            adj_indices[b, u, v] = b_type
                            adj_indices[b, v, u] = b_type
                            adj_tensor[b, u, v, :] = 0.0
                            adj_tensor[b, v, u, :] = 0.0
                            adj_tensor[b, u, v, b_type] = 1.0
                            adj_tensor[b, v, u, b_type] = 1.0
                        else:
                            adj_indices[b, u, v] = 0
                            adj_indices[b, v, u] = 0
                            adj_tensor[b, u, v, :] = 0.0
                            adj_tensor[b, v, u, :] = 0.0
                            adj_tensor[b, u, v, 0] = 1.0
                            adj_tensor[b, v, u, 0] = 1.0

                # Ensure active node u is connected to at least one prior active node
                for b in range(batch_size):
                    if step < target_sizes[b] and adj_indices[b, u, :u].sum() == 0 and u > 0:
                        u_t = int(node_indices[b, u].item())
                        candidates = [c for c in range(u) if c < target_sizes[b]]
                        candidates.sort(key=lambda c: 0 if int(node_indices[b, c].item()) == 1 else 1)
                        for v_cand in candidates:
                            v_type = int(node_indices[b, v_cand].item())
                            if u_t == 3 and v_type == 3:
                                continue
                            if (u_t == 4 and v_type != 1) or (v_type == 4 and u_t != 1):
                                continue
                            if u_t == 2 and v_type == 2:
                                u_has_n = any(node_indices[b, w].item() == 2 and adj_indices[b, u, w].item() > 0 for w in range(step) if w != u and w != v_cand)
                                v_has_n = any(node_indices[b, w].item() == 2 and adj_indices[b, v_cand, w].item() > 0 for w in range(step) if w != u and w != v_cand)
                                if u_has_n or v_has_n:
                                    continue
                            curr_val_u = sum(BOND_ORDERS[adj_indices[b, u, w].item()] for w in range(step))
                            curr_val_v = sum(BOND_ORDERS[adj_indices[b, v_cand, w].item()] for w in range(step))
                            if (curr_val_u + 1.0 <= MAX_VALENCES[u_t]) and (curr_val_v + 1.0 <= MAX_VALENCES[v_type]):
                                adj_indices[b, u, v_cand] = 1
                                adj_indices[b, v_cand, u] = 1
                                adj_tensor[b, u, v_cand, :] = 0.0
                                adj_tensor[b, v_cand, u, :] = 0.0
                                adj_tensor[b, u, v_cand, 1] = 1.0
                                adj_tensor[b, v_cand, u, 1] = 1.0
                                break

            trajectories.append(step_record)

        return node_indices, adj_indices, node_features, adj_tensor, trajectories

    def evaluate_trajectory_log_probs(self, trajectories, temperature=1.2, z_cond=None):
        """Re-computes action log probabilities and entropy across recorded trajectories."""
        total_log_probs = 0.0
        entropies = []

        for record in trajectories:
            step = record['step']
            node_feat = record['state_node']
            adj_t = record['state_adj']
            mask = record['active_mask']
            device = node_feat.device
            batch_size = node_feat.size(0)

            node_emb, graph_emb = self.forward_subgraph(node_feat, adj_t, mask, z_cond=z_cond)

            # Node decision evaluation
            bond_angles = record['bond_angles']
            free_val = record['free_val']
            neighbor_types = record['neighbor_types']

            q_logits, _ = self.quantum_policy(graph_emb, bond_angles, free_val, neighbor_types)
            if 'node_mask' in record:
                q_logits = torch.where(record['node_mask'], q_logits, torch.tensor(-1e9, device=device))
            elif step == 0:
                q_logits[:, 1:] = -1e9

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            total_log_probs = total_log_probs + node_dist.log_prob(record['node_action'])
            entropies.append(node_dist.entropy())

            # Edge decisions evaluation
            for edge_rec in record['edge_records']:
                u = edge_rec['u']
                v = edge_rec['v']
                pair_feat = torch.cat([node_emb[:, u, :], node_emb[:, v, :], graph_emb], dim=-1)
                edge_logits = self.edge_head(pair_feat)
                edge_logits = torch.where(edge_rec['valence_mask'], edge_logits, torch.tensor(-1e9, device=device))

                edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                total_log_probs = total_log_probs + edge_dist.log_prob(edge_rec['edge_action'])
                entropies.append(edge_dist.entropy())

        mean_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
        return total_log_probs, mean_entropy


class CriticRGCN(nn.Module):
    """Value Critic evaluating intermediate and final molecular graph quality."""
    def __init__(self, node_dim=5, edge_types=5, hidden_dims=[128, 64], dropout_rate=0.0, z_dim=32):
        super(CriticRGCN, self).__init__()
        self.encoder = RelationalGCN(node_dim, edge_types, hidden_dims, dropout_rate)
        self.z_adapter = nn.Sequential(
            nn.Linear(z_dim, self.encoder.out_dim),
            nn.LeakyReLU(0.2)
        )
        self.val_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_tensor, node_mask=None, z_cond=None):
        node_emb, graph_emb = self.encoder(node_features, adj_tensor, node_mask=node_mask)
        if z_cond is not None:
            graph_emb = graph_emb + self.z_adapter(z_cond)
        return self.val_head(graph_emb)


class DiscriminatorRGCN(nn.Module):
    """Discriminator network judging molecular realism."""
    def __init__(self, node_dim=5, edge_types=5, hidden_dims=[128, 64], dropout_rate=0.0):
        super(DiscriminatorRGCN, self).__init__()
        self.encoder = RelationalGCN(node_dim, edge_types, hidden_dims, dropout_rate)
        self.disc_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_tensor, node_mask=None):
        node_emb, graph_emb = self.encoder(node_features, adj_tensor, node_mask=node_mask)
        return self.disc_head(graph_emb)
