import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from molgan_vqp.layers import RelationalGCN
from molgan_vqp.policy_qc import (
    JerbiAtomPolicy,
    JerbiEdgePolicy,
    DedicatedAtomQuantumPolicy,
    DedicatedBondQuantumPolicy,
    QuantumEdgePolicy2Q,
    ClassicalEdgeCircuit
)

# ─────────────────────────────────────────────────────────────────────────────
# CNS 9-Element Atom Vocabulary (Adapted from MolGAN-QPolicy-Alzheimers Phase 2)
# Index 0 = PAD (no atom); indices 1–9 = real heavy atoms
#   1: C   2: N   3: O   4: F   5: I   6: P   7: S   8: Cl  9: Br
# ─────────────────────────────────────────────────────────────────────────────

# Maximum chemical valence per atom-type index
MAX_VALENCES = {
    0: 0,   # PAD
    1: 4,   # C
    2: 3,   # N
    3: 2,   # O
    4: 1,   # F   (monovalent halogen)
    5: 1,   # I   (monovalent halogen)
    6: 5,   # P   (tetrahedral / pentavalent)
    7: 2,   # S   (thioether valence 2)
    8: 1,   # Cl  (monovalent halogen)
    9: 1,   # Br  (monovalent halogen)
}

# Atom-type indices that are strictly monovalent terminal halogens
HALOGEN_TYPES = frozenset({4, 5, 8, 9})   # F, I, Cl, Br

# Bond orders matching 4 bond types (AROMATIC BOND REMOVED):
# 0: None, 1: Single (1.0), 2: Double (2.0), 3: Triple (3.0)
BOND_ORDERS = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0
}


def _shortest_path_dist(adj_b, start, target, max_u, max_dist=8):
    """
    Computes shortest path distance between start and target in the subgraph on [0, max_u).
    Returns path length (in edges) or 999 if unreachable or > max_dist.
    """
    if start == target:
        return 0
    visited = {start}
    queue = [(start, 0)]
    while queue:
        curr, d = queue.pop(0)
        if d >= max_dist:
            return 999
        for nbr in range(max_u):
            if adj_b[curr, nbr].item() > 0 and nbr not in visited:
                if nbr == target:
                    return d + 1
                visited.add(nbr)
                queue.append((nbr, d + 1))
    return 999


def _get_shortest_path_nodes(adj_b, start, target, max_u, max_dist=8):
    """Returns list of nodes on shortest path from start to target, or None."""
    if start == target:
        return [start]
    visited = {start: None}
    queue = [start]
    while queue:
        curr = queue.pop(0)
        if curr == target:
            path = []
            c = target
            while c is not None:
                path.append(c)
                c = visited[c]
            path.reverse()
            return path if len(path) <= max_dist + 1 else None
        for nbr in range(max_u):
            if adj_b[curr, nbr].item() > 0 and nbr not in visited:
                visited[nbr] = curr
                queue.append(nbr)
    return None


def _node_in_any_cycle(adj_b, node, max_u):
    """Checks if node is already part of any cycle in subgraph [0, max_u)."""
    nbrs = [nbr for nbr in range(max_u) if adj_b[node, nbr].item() > 0]
    if len(nbrs) < 2:
        return False
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            visited = {node, nbrs[i]}
            q = [nbrs[i]]
            while q:
                curr = q.pop(0)
                if curr == nbrs[j]:
                    return True
                for nxt in range(max_u):
                    if adj_b[curr, nxt].item() > 0 and nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
    return False


def _in_same_cycle(adj_b, w, v, max_u):
    """
    Checks if vertices w and v in [0, max_u) already lie on the same cycle
    (i.e., there exist >= 2 edge-disjoint paths between w and v).
    Connecting a new vertex u to both w and v would bridge across that cycle,
    creating an unstable bridged bicycle / cage with bridgehead atoms.
    """
    if w == v:
        return False
    visited = {w: None}
    queue = [w]
    found = False
    while queue:
        curr = queue.pop(0)
        if curr == v:
            found = True
            break
        for nbr in range(max_u):
            if adj_b[curr, nbr].item() > 0 and nbr not in visited:
                visited[nbr] = curr
                queue.append(nbr)
    if not found:
        return False

    # Trace shortest path from w to v
    path = []
    curr = v
    while curr is not None:
        path.append(curr)
        curr = visited[curr]
    path.reverse()
    if len(path) < 2:
        return False

    # Check if alternate path exists without edge (path[0], path[1])
    e0, e1 = path[0], path[1]
    visited2 = {w}
    q2 = [w]
    while q2:
        curr = q2.pop(0)
        if curr == v:
            return True
        for nbr in range(max_u):
            if (curr == e0 and nbr == e1) or (curr == e1 and nbr == e0):
                continue
            if adj_b[curr, nbr].item() > 0 and nbr not in visited2:
                visited2.add(nbr)
                q2.append(nbr)
    return False


class ActorRGCN_Jerbi(nn.Module):
    """
    Autoregressive Graph Convolutional Actor with Jerbi et al. (NeurIPS 2021)
    Variational Quantum Policies (Softmax-PQC with Multi-Layer Data Re-Uploading Rotation Policy):
      - Atom Selection: 9-Qubit Dedicated Jerbi PQC Head (1 Qubit = 1 Heavy Atom: C, N, O, F, I, P, S, Cl, Br)
      - Bond Formation: 4-Qubit Dedicated Jerbi PQC Head (or MLP/Control Edge Head)
      - Feature-wise Linear Modulation (FiLM) Conditioning from Phase 1 Graph VAE latent codes z ~ N(0, I) in R^32:
          g_cond = (1 + tanh(gamma(z))) * g + beta(z)
      - Strict Chemical Valence & Anti-Strain Rules (Bredt's Rule, Anti-Cycloalkyne, Anti-Thione, Anti-Cage).
    """

    def __init__(self, max_atoms=35, min_atoms=9, num_atom_types=9, num_edge_types=4,
                 node_dim=10, hidden_dims=None, dropout_rate=0.0, z_dim=32,
                 n_layers_atom=3, n_layers_edge=3, output_scale=3.0,
                 atom_policy_type='jerbi', edge_policy_type='mlp',
                 use_valence_mask=True, enforce_connectivity=True, use_atom_filters=True):
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super(ActorRGCN_Jerbi, self).__init__()
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.num_atom_types = num_atom_types   # 9 real atom types (or 4 for QM9)
        self.num_edge_types = num_edge_types   # 4 bond types (None, Single, Double, Triple)
        self.node_dim = node_dim               # 10 (PAD + 9 atom types)
        self.z_dim = z_dim
        self.atom_policy_type = str(atom_policy_type).lower()
        self.edge_policy_type = str(edge_policy_type).lower()
        self.use_valence_mask = use_valence_mask
        self.enforce_connectivity = enforce_connectivity
        self.use_atom_filters = use_atom_filters
        self.n_layers_atom = n_layers_atom
        self.n_layers_edge = n_layers_edge
        self.output_scale = output_scale

        # Relational GCN encoder
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=num_edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        # Feature-wise Linear Modulation (FiLM) Adapter for Phase 1 VAE prior z in R^32
        self.z_adapter = nn.Sequential(
            nn.Linear(z_dim, self.encoder.out_dim * 2),
            nn.LeakyReLU(0.2)
        )

        # 1. ATOM SELECTION POLICY HEAD (Jerbi Data Re-Uploading PQC Head)
        self.quantum_policy = JerbiAtomPolicy(
            in_features=self.encoder.out_dim,
            n_qubits=num_atom_types,
            n_layers=n_layers_atom,
            output_scale=output_scale,
            learnable_scale=True
        )
        self.atom_policy = self.quantum_policy

        # 2. BOND FORMATION POLICY HEAD
        pair_in_dim = self.encoder.out_dim * 3
        if self.edge_policy_type in ['jerbi', 'vqp', 'quantum']:
            self.edge_policy = JerbiEdgePolicy(
                in_features=pair_in_dim,
                n_layers=n_layers_edge,
                output_scale=output_scale,
                learnable_scale=True
            )
        elif self.edge_policy_type in ['dedicated', 'dedicated_quantum', '4q']:
            self.edge_policy = DedicatedBondQuantumPolicy(
                in_features=pair_in_dim,
                n_layers=n_layers_edge,
                output_scale=output_scale
            )
        elif self.edge_policy_type in ['quantum_2q', '2q', 'vqc']:
            self.edge_policy = QuantumEdgePolicy2Q(
                in_features=pair_in_dim,
                n_layers=n_layers_edge,
                output_scale=output_scale,
                learnable_scale=True
            )
        elif self.edge_policy_type in ['classical', 'size_matched']:
            self.edge_policy = ClassicalEdgeCircuit(
                in_features=pair_in_dim,
                hidden_dim=8,
                num_classes=num_edge_types
            )
        elif self.edge_policy_type in ['mlp', 'direct']:
            self.edge_policy = nn.Sequential(
                nn.Linear(pair_in_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_edge_types)
            )
        else:
            self.edge_policy = nn.Sequential(
                nn.Linear(pair_in_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_edge_types)
            )

        self.edge_head = self.edge_policy

    def forward_subgraph(self, node_features, adj_tensor, active_mask, z_cond=None):
        """Encodes active subgraph with FiLM modulation from Phase 1 VAE latent prior z."""
        node_emb, graph_emb = self.encoder(node_features, adj_tensor, node_mask=active_mask)
        if z_cond is not None:
            film = self.z_adapter(z_cond)
            gamma, beta = torch.chunk(film, 2, dim=-1)
            graph_emb = (1.0 + torch.tanh(gamma)) * graph_emb + beta
        return node_emb, graph_emb

    def forward_atom(self, graph_emb, bond_angle=None, free_valence=None, neighbor_type=None):
        """Invokes Jerbi Atom Policy."""
        return self.quantum_policy(graph_emb, bond_angle, free_valence, neighbor_type)

    def forward_edge(self, pair_feat):
        """Invokes Bond Policy."""
        out = self.edge_head(pair_feat)
        if isinstance(out, tuple):
            return out[0]
        return out

    def compute_valence_mask(self, node_indices, adj_indices, u, v):
        """
        Dynamically computes legal bond types between atom u and atom v.
        Enforces 14 comprehensive chemical valence, topology & anti-strain rules:
          1. PAD atoms cannot bond
          2. Monovalent halogens (F, Cl, Br, I single-bond to Carbon only)
          3. Anti-geminal mixed halogen clusters (bans -C(F)Cl, max -CF2)
          4. Anti-peroxide check (O-O banned)
          5. Anti-hydrazine / azo check (N-N banned)
          6. Anti-hydroxylamine check (N-O banned)
          7. Anti-disulfide check (S-S banned)
          8. Anti-thione rule (Sulfur restricted to single bonds, no C=S)
          9. Valence budget calculation
          10. Anti-poly-yne rule (at most 1 triple bond)
          11. Anti-cumulene rule (no adjacent double bonds)
          12. Anti-cycloalkyne rule (triple bonds strictly banned in cycles)
          13. Dynamic Bredt's rule (double bonds banned at small-ring bridgeheads)
          14. Strict ring allowlist (only unstrained 5-, 6-, 7-membered rings)
        """
        batch_size = node_indices.size(0)
        device = node_indices.device
        mask = torch.ones(batch_size, self.num_edge_types, dtype=torch.bool, device=device)

        for b in range(batch_size):
            tu = int(node_indices[b, u].item())
            tv = int(node_indices[b, v].item())

            # 1. PAD atoms cannot bond
            if tu == 0 or tv == 0:
                mask[b, 1:] = False
                continue

            # 2. Halogen rules: F/I/Cl/Br are monovalent and only single-bond to Carbon
            if tu in HALOGEN_TYPES or tv in HALOGEN_TYPES:
                if not ((tu in HALOGEN_TYPES and tv == 1) or
                        (tv in HALOGEN_TYPES and tu == 1)):
                    mask[b, 1:] = False
                    continue
                # Single bond only (no double or triple)
                mask[b, 2:] = False

            # 3. Anti-geminal mixed halogen rule: prevent messy -C(F)Cl or mixed halogens on same carbon
            if tu in HALOGEN_TYPES:
                v_halogens = [
                    int(node_indices[b, w].item()) for w in range(u)
                    if adj_indices[b, v, w].item() > 0 and int(node_indices[b, w].item()) in HALOGEN_TYPES
                ]
                if len(v_halogens) >= 1:
                    # If carbon v already has a halogen, only allow F if existing is also F (for clean -CF2)
                    if tu != 4 or any(h != 4 for h in v_halogens) or len(v_halogens) >= 2:
                        mask[b, 1:] = False
                        continue

            # 4. Anti-peroxide: O-O bond banned
            if tu == 3 and tv == 3:
                mask[b, 1:] = False
                continue

            # 5. Anti-hydrazine / azo: N-N bond banned
            if tu == 2 and tv == 2:
                mask[b, 1:] = False
                continue

            # 6. Anti-hydroxylamine: N-O bond banned
            if (tu == 2 and tv == 3) or (tu == 3 and tv == 2):
                mask[b, 1:] = False
                continue

            # 7. Anti-disulfide: S-S banned completely in CNS small-molecule drugs
            if tu == 7 and tv == 7:
                mask[b, 1:] = False
                continue

            # 8. Anti-thione rule: Sulfur forms single bonds only (thioether/thiazole), no C=S double bonds
            if tu == 7 or tv == 7:
                mask[b, 2:] = False

            # 9. Valence budget
            curr_val_u = sum(BOND_ORDERS[adj_indices[b, u, w].item()] for w in range(u) if w != v)
            curr_val_v = sum(BOND_ORDERS[adj_indices[b, v, w].item()] for w in range(u) if w != v)
            rem_u = MAX_VALENCES.get(tu, 1) - curr_val_u
            rem_v = MAX_VALENCES.get(tv, 1) - curr_val_v

            for bt in range(1, self.num_edge_types):
                if BOND_ORDERS[bt] > rem_u or BOND_ORDERS[bt] > rem_v:
                    mask[b, bt] = False

            # 10. Anti-poly-yne: at most 1 triple bond per molecule
            if self.num_edge_types > 3:
                num_triple = sum(
                    1 for w1 in range(u)
                    for w2 in range(w1 + 1, u)
                    if adj_indices[b, w1, w2].item() == 3
                )
                if num_triple >= 1:
                    mask[b, 3] = False

            # 11. Anti-cumulene: prevent adjacent double bonds on same atom
            if self.num_edge_types > 2:
                u_has_double = any(adj_indices[b, u, w].item() == 2 for w in range(u) if w != v)
                v_has_double = any(adj_indices[b, v, w].item() == 2 for w in range(u) if w != u)
                if u_has_double or v_has_double:
                    mask[b, 2] = False

            # 12. Cap backward connections per atom (Anti-Cage)
            u_bonds_to_prev = sum(1 for w in range(u) if adj_indices[b, u, w].item() > 0)
            if u_bonds_to_prev >= 2:
                mask[b, 1:] = False
                continue

            # Cap total rings per molecule using graph cyclomatic number: Rings = E - V + 1
            if u_bonds_to_prev >= 1:
                n_active_prev = sum(1 for w in range(u) if int(node_indices[b, w].item()) > 0)
                n_edges_prev = sum(1 for w1 in range(u) for w2 in range(w1 + 1, u) if adj_indices[b, w1, w2].item() > 0)
                curr_rings = max(0, n_edges_prev - n_active_prev + 1)
                if curr_rings >= 2:
                    mask[b, 1:] = False
                    continue

            # Heavy Degree Cap on target atom v
            v_heavy_deg = sum(1 for w in range(u) if adj_indices[b, v, w].item() > 0 and w != u)
            if tv == 1 and v_heavy_deg >= 3:
                mask[b, 1:] = False
                continue
            elif tv in (2, 7) and v_heavy_deg >= 2:
                mask[b, 1:] = False
                continue
            elif tv == 3 and v_heavy_deg >= 2:
                mask[b, 1:] = False
                continue
            elif tv in HALOGEN_TYPES and v_heavy_deg >= 1:
                mask[b, 1:] = False
                continue

            # 13. Dynamic Bredt's Rule & Anti-Cycloalkyne Enforcement
            # Rule A: If atom u is already bonded to a previous atom (u_bonds_to_prev >= 1),
            # any additional connection to v closes a ring. ALL ring closures MUST be single bonds!
            if u_bonds_to_prev >= 1:
                mask[b, 2:] = False
                # If target v already has a double bond, closing a ring on v makes v a bridgehead double bond (Bredt violation)
                if any(adj_indices[b, v, w].item() == 2 for w in range(u) if w != u):
                    mask[b, 1:] = False
                    continue
                # If u already has a double bond, u cannot close a ring (Bredt violation)
                if any(adj_indices[b, u, w].item() == 2 for w in range(u)):
                    mask[b, 1:] = False
                    continue

            # Rule B: If atom v already has >= 2 bonds (v_heavy_deg >= 2), v is a ring junction
            # or bridgehead candidate. Bonds to v MUST be single bonds!
            if v_heavy_deg >= 2:
                mask[b, 2:] = False

            # Rule C: If atom u already has a double bond, u cannot form a second backward bond
            if any(adj_indices[b, u, w].item() == 2 for w in range(u)):
                mask[b, 1:] = False
                continue

            # Rule D: If atom v already has a double bond, no double bond allowed to v
            if any(adj_indices[b, v, w].item() == 2 for w in range(u) if w != u):
                mask[b, 2] = False

            # 14. Strict Ring Size Allowlist (ONLY unstrained 5, 6, 7-Membered Rings; ZERO 3, 4, or > 7)
            if u_bonds_to_prev >= 1:
                u_neighbors = [w for w in range(u) if adj_indices[b, u, w].item() > 0]
                legal_ring = True
                for w in u_neighbors:
                    # Anti-Bridged Cage Rule: If w and v are already in the same cycle,
                    # connecting u to both w and v bridges across that cycle, creating
                    # an unstable bridged bicycle (CalcNumBridgeheadAtoms > 0 and Bredt violation)!
                    if _in_same_cycle(adj_indices[b], w, v, u):
                        legal_ring = False
                        break

                    path = _get_shortest_path_nodes(adj_indices[b], w, v, u, max_dist=7)
                    if path is None:
                        legal_ring = False
                        break

                    # A. Anti-Cycloalkyne: NO triple bonds anywhere in the closed cycle!
                    if any(adj_indices[b, path[i], path[i+1]].item() == 3 for i in range(len(path)-1)):
                        legal_ring = False
                        break

                    # B. Strict Ring Size: ring size is len(path) + 1 (including vertex u)
                    # Only 5, 6, 7-membered rings allowed (path length 4, 5, 6)
                    if len(path) not in (4, 5, 6):
                        legal_ring = False
                        break

                    # C. Anti-Bridged Bicycle: Count how many nodes on path are already in cycles
                    # In fused rings (e.g. naphthalene, indole), rings share at most 2 adjacent nodes (1 edge).
                    # If path shares >= 3 nodes with existing cycles, it creates a bridged bicycle with bridgeheads!
                    cycle_nodes_on_path = [node for node in path if _node_in_any_cycle(adj_indices[b], node, u)]
                    if len(cycle_nodes_on_path) > 2:
                        legal_ring = False
                        break

                    if self.num_edge_types > 3:
                        mask[b, 3] = False

                if not legal_ring:
                    mask[b, 1:] = False
                    continue

        return mask

    def generate_mols(self, batch_size, device='cpu', temperature=1.2,
                      min_atoms=9, max_atoms=None, target_sizes=None, z_cond=None):
        """
        Autoregressively generates molecules with flexible sizes.
        Uses Jerbi Data Re-Uploading Softmax-PQC for atom and bond decisions.
        Conditioned on Phase 1 Graph VAE latent codes z ~ N(0, I) in R^32 with FiLM modulation.
        """
        actual_max = max_atoms if max_atoms is not None else self.max_atoms

        if target_sizes is None:
            target_sizes = torch.randint(min_atoms, actual_max + 1, (batch_size,), device=device)
        elif isinstance(target_sizes, int):
            target_sizes = torch.full((batch_size,), target_sizes, dtype=torch.long, device=device)
        else:
            target_sizes = target_sizes.to(device)

        if z_cond is None:
            z_cond = torch.randn(batch_size, self.z_dim, device=device)
        else:
            z_cond = z_cond.to(device)

        node_features = torch.zeros(batch_size, actual_max, self.node_dim, device=device)
        node_features[:, :, 0] = 1.0   # PAD
        node_indices = torch.zeros((batch_size, actual_max), dtype=torch.long, device=device)

        adj_tensor = torch.zeros(batch_size, actual_max, actual_max, self.num_edge_types, device=device)
        adj_tensor[:, :, :, 0] = 1.0   # No bond
        adj_indices = torch.zeros(batch_size, actual_max, actual_max, dtype=torch.long, device=device)

        active_mask = torch.zeros(batch_size, actual_max, device=device)
        trajectories = []

        atom_counts = {t: torch.zeros(batch_size, device=device) for t in range(1, self.num_atom_types + 1)}

        for step in range(actual_max):
            active_step_mask = (step < target_sizes)
            step_record = {
                'step': step,
                'state_node': node_features.clone(),
                'state_adj': adj_tensor.clone(),
                'active_mask': active_mask.clone(),
                'active_batch_mask': active_step_mask.clone(),
                'edge_records': []
            }

            node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask, z_cond=z_cond)

            # Local geometric descriptors for quantum policy
            if step == 0:
                bond_angles = torch.full((batch_size,), 1.911, device=device)
                free_val = torch.full((batch_size,), 4.0, device=device)
                neighbor_types = torch.ones(batch_size, dtype=torch.long, device=device)
            else:
                bond_angles = torch.full((batch_size,), 1.911, device=device)
                free_val = torch.full((batch_size,), 2.0, device=device)
                last_atom = node_indices[:, step - 1].clamp(min=1)
                neighbor_types = last_atom

            q_logits, _ = self.forward_atom(graph_emb, bond_angles, free_val, neighbor_types)

            # Masking for step 0 and chemical priors
            node_mask = torch.ones(batch_size, self.num_atom_types, dtype=torch.bool, device=device)
            if step == 0 and self.use_atom_filters:
                # Force Carbon anchor at step 0
                node_mask[:, 1:] = False
            elif self.use_atom_filters:
                # Apply strict CNS medicinal chemistry caps for real drug-likeness
                for b in range(batch_size):
                    t_size = target_sizes[b].item()

                    # 1. Hard ban exotic elements in small-molecule CNS therapeutics:
                    if self.num_atom_types > 4:
                        node_mask[b, 4] = False  # Iodine (I)
                    if self.num_atom_types > 5:
                        node_mask[b, 5] = False  # Phosphorus (P)

                    # 2. Halogen exclusivity and realistic medicinal caps:
                    f_count = atom_counts[4][b].item() if 4 in atom_counts else 0
                    cl_count = atom_counts[8][b].item() if 8 in atom_counts else 0
                    br_count = atom_counts[9][b].item() if 9 in atom_counts else 0

                    if self.num_atom_types > 8:
                        if br_count >= 1 or f_count > 0 or cl_count > 0:
                            node_mask[b, 8] = False  # Bromine (Br)

                    if self.num_atom_types > 7:
                        if cl_count >= 1 or br_count > 0:
                            node_mask[b, 7] = False  # Chlorine (Cl)

                    if self.num_atom_types > 3:
                        if f_count >= 3 or br_count > 0:
                            node_mask[b, 3] = False  # Fluorine (F)

                    # 3. Sulfur cap: Max 1 atom
                    if self.num_atom_types > 6:
                        s_count = atom_counts[7][b].item() if 7 in atom_counts else 0
                        if s_count >= 1:
                            node_mask[b, 6] = False

                    # 4. Nitrogen & Oxygen caps
                    n_count = atom_counts[2][b].item() if 2 in atom_counts else 0
                    if n_count >= min(4, math.ceil(0.22 * t_size)):
                        node_mask[b, 1] = False

                    o_count = atom_counts[3][b].item() if 3 in atom_counts else 0
                    if o_count >= min(4, math.ceil(0.20 * t_size)):
                        node_mask[b, 2] = False

                    # 5. Carbon backbone enforcement (real CNS drugs are 65% - 85% Carbon)
                    s_count = atom_counts[7][b].item() if 7 in atom_counts else 0
                    tot_het = n_count + o_count + f_count + cl_count + br_count + s_count
                    max_het_budget = max(2, int(math.floor(0.35 * t_size)))
                    if tot_het >= max_het_budget:
                        node_mask[b, 1:] = False

                    c_count = atom_counts[1][b].item() if 1 in atom_counts else 0
                    if step >= 3 and (c_count / step) < 0.65:
                        node_mask[b, 1:] = False

            q_logits = torch.where(node_mask, q_logits, torch.tensor(-1e9, device=device))

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            node_action = node_dist.sample()
            node_log_prob = node_dist.log_prob(node_action)

            atom_types = node_action + 1  # Action 0..8 -> Types 1..9

            step_record['node_action'] = node_action
            step_record['node_log_prob'] = node_log_prob
            step_record['node_mask'] = node_mask.clone()
            step_record['bond_angles'] = bond_angles
            step_record['free_val'] = free_val
            step_record['neighbor_types'] = neighbor_types

            for b in range(batch_size):
                if active_step_mask[b]:
                    a_type = atom_types[b].item()
                    node_indices[b, step] = a_type
                    node_features[b, step, :] = 0.0
                    node_features[b, step, a_type] = 1.0
                    active_mask[b, step] = 1.0
                    atom_counts[a_type][b] += 1
                else:
                    node_indices[b, step] = 0
                    node_features[b, step, :] = 0.0
                    node_features[b, step, 0] = 1.0
                    active_mask[b, step] = 0.0

            # Edge formation for new atom with existing atoms
            if step > 0:
                node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask, z_cond=z_cond)
                u = step
                for v in range(step):
                    h_u = node_emb[:, u, :]
                    h_v = node_emb[:, v, :]
                    pair_feat = torch.cat([h_u, h_v, graph_emb], dim=-1)

                    edge_logits = self.forward_edge(pair_feat)

                    if self.use_valence_mask:
                        valence_mask = self.compute_valence_mask(node_indices, adj_indices, u, v)
                        edge_logits = torch.where(valence_mask, edge_logits, torch.tensor(-1e9, device=device))

                        # Medicinal Ring Formation Incentive (+2.5 boost for unstrained 5/6 rings)
                        for b in range(batch_size):
                            if active_step_mask[b] and valence_mask[b, 1]:
                                u_prev_bonds = [w for w in range(u) if adj_indices[b, u, w].item() > 0]
                                if len(u_prev_bonds) == 1:
                                    w = u_prev_bonds[0]
                                    d = _shortest_path_dist(adj_indices[b], w, v, u, max_dist=8)
                                    if d in (3, 4):  # 5-membered (d=3) or 6-membered (d=4) rings
                                        edge_logits[b, 1] += 2.5
                    else:
                        valence_mask = None

                    edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                    edge_action = edge_dist.sample()
                    edge_log_prob = edge_dist.log_prob(edge_action)

                    step_record['edge_records'].append({
                        'u': u,
                        'v': v,
                        'edge_action': edge_action,
                        'edge_log_prob': edge_log_prob,
                        'valence_mask': valence_mask.clone() if valence_mask is not None else None
                    })

                    for b in range(batch_size):
                        if active_step_mask[b]:
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

                # Connectivity enforcement with strict valence preservation
                if self.enforce_connectivity:
                    for b in range(batch_size):
                        if active_step_mask[b] and adj_indices[b, u, :u].sum() == 0:
                            u_t = int(node_indices[b, u].item())
                            if u_t == 0:
                                continue
                            rem_u = MAX_VALENCES.get(u_t, 1)
                            connected = False
                            for v_cand in range(u - 1, -1, -1):
                                v_t = int(node_indices[b, v_cand].item())
                                if v_t == 0:
                                    continue
                                if (u_t in HALOGEN_TYPES or v_t in HALOGEN_TYPES) and not (
                                    (u_t in HALOGEN_TYPES and v_t == 1) or (v_t in HALOGEN_TYPES and u_t == 1)
                                ):
                                    continue
                                v_deg = sum(1 for w in range(u) if adj_indices[b, v_cand, w].item() > 0)
                                if v_t == 1 and v_deg >= 3:
                                    continue
                                elif v_t in (2, 7) and v_deg >= 2:
                                    continue
                                elif v_t == 3 and v_deg >= 2:
                                    continue
                                elif v_t in HALOGEN_TYPES:
                                    continue

                                val_v = sum(BOND_ORDERS[adj_indices[b, v_cand, w].item()] for w in range(u) if w != v_cand)
                                rem_v = MAX_VALENCES.get(v_t, 1) - val_v
                                if rem_v >= 1.0 and rem_u >= 1.0:
                                    adj_indices[b, u, v_cand] = 1
                                    adj_indices[b, v_cand, u] = 1
                                    adj_tensor[b, u, v_cand, :] = 0.0
                                    adj_tensor[b, v_cand, u, :] = 0.0
                                    adj_tensor[b, u, v_cand, 1] = 1.0
                                    adj_tensor[b, v_cand, u, 1] = 1.0
                                    connected = True
                                    break
                            if not connected:
                                node_indices[b, u] = 0
                                node_features[b, u, :] = 0.0
                                node_features[b, u, 0] = 1.0
                                active_mask[b, u] = 0.0
                                if u_t in atom_counts:
                                    atom_counts[u_t][b] = max(0, atom_counts[u_t][b] - 1)

            trajectories.append(step_record)

        return node_indices, adj_indices, node_features, adj_tensor, trajectories

    def evaluate_trajectory_log_probs(self, trajectories, temperature=1.2, z_cond=None):
        """Re-computes action log probabilities and entropy across recorded trajectories."""
        if not trajectories:
            return torch.tensor(0.0), torch.tensor(0.0)

        device = trajectories[0]['state_node'].device
        batch_size = trajectories[0]['state_node'].size(0)
        total_log_probs = torch.zeros(batch_size, device=device)
        entropies = []

        for record in trajectories:
            step = record['step']
            node_feat = record['state_node']
            adj_t = record['state_adj']
            mask = record['active_mask']
            abm = record.get('active_batch_mask', None)
            if abm is None:
                abm = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                abm = abm.to(device)

            if not abm.any():
                continue

            node_emb, graph_emb = self.forward_subgraph(node_feat, adj_t, mask, z_cond=z_cond)

            bond_angles = record['bond_angles']
            free_val = record['free_val']
            neighbor_types = record['neighbor_types']

            q_logits, _ = self.forward_atom(graph_emb, bond_angles, free_val, neighbor_types)
            node_mask = record['node_mask']
            q_logits = torch.where(node_mask, q_logits, torch.tensor(-1e9, device=device))

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            node_act = record['node_action']
            node_lp = node_dist.log_prob(node_act)
            node_lp = torch.where(abm, node_lp, torch.zeros_like(node_lp))
            total_log_probs = total_log_probs + node_lp

            if abm.any():
                entropies.append(node_dist.entropy()[abm].mean())

            for e_rec in record['edge_records']:
                u, v = e_rec['u'], e_rec['v']
                edge_act = e_rec['edge_action']
                val_mask = e_rec['valence_mask']

                pair_feat = torch.cat([node_emb[:, u, :], node_emb[:, v, :], graph_emb], dim=-1)
                edge_logits = self.forward_edge(pair_feat)

                if self.use_valence_mask and val_mask is not None:
                    edge_logits = torch.where(val_mask, edge_logits, torch.tensor(-1e9, device=device))

                edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                edge_lp = edge_dist.log_prob(edge_act)
                edge_lp = torch.where(abm, edge_lp, torch.zeros_like(edge_lp))
                total_log_probs = total_log_probs + edge_lp

                if abm.any():
                    entropies.append(edge_dist.entropy()[abm].mean())

        mean_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=device)
        return total_log_probs, mean_entropy


# Alias ActorRGCN_Quantum for seamless compatibility with final phase 2
ActorRGCN_Quantum = ActorRGCN_Jerbi


# =============================================================================
# 2. VALUE CRITIC NETWORK V_psi(G, z)
# =============================================================================

class CriticRGCN(nn.Module):
    """Value Critic V_psi(G, z) predicting expected cumulative return with FiLM conditioning."""
    def __init__(self, node_dim=10, edge_types=4, z_dim=32, hidden_dims=None, dropout_rate=0.0):
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super(CriticRGCN, self).__init__()
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        self.z_dim = z_dim
        self.z_adapter = nn.Sequential(
            nn.Linear(z_dim, self.encoder.out_dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_tensor, node_mask=None, z_cond=None):
        _, graph_emb = self.encoder(node_features, adj_tensor, node_mask=node_mask)
        if z_cond is not None:
            film = self.z_adapter(z_cond)
            gamma, beta = torch.chunk(film, 2, dim=-1)
            graph_emb = (1.0 + torch.tanh(gamma)) * graph_emb + beta
        return self.value_head(graph_emb)


# =============================================================================
# 3. ADVERSARIAL DISCRIMINATOR NETWORK D_phi(G)
# =============================================================================

class DiscriminatorRGCN(nn.Module):
    """Adversarial Discriminator D_phi(G) evaluating realism of real vs generated graphs."""
    def __init__(self, node_dim=10, edge_types=4, hidden_dims=None, dropout_rate=0.1):
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super(DiscriminatorRGCN, self).__init__()
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        self.disc_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_tensor, node_mask=None):
        _, graph_emb = self.encoder(node_features, adj_tensor, node_mask=node_mask)
        return self.disc_head(graph_emb)
