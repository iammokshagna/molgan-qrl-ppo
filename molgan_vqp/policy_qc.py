import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# =============================================================================
# 1. JERBI MULTI-LAYER DATA RE-UPLOADING PARAMETRIZED QUANTUM POLICY HEAD
#    Based on Jerbi et al. (NeurIPS 2021) "Parametrized Quantum Policies for RL"
# =============================================================================

class JerbiDataReuploadingPQC(nn.Module):
    """
    Parametrized Quantum Policy Head based on Jerbi et al. (NeurIPS 2021):
    'Parametrized Quantum Policies for Reinforcement Learning' (arXiv:2103.05577).

    Key Characteristics:
      1. Multi-Layer Data Re-Uploading:
         The classical state representation s is repeatedly re-injected across all L layers
         via single-qubit rotations, expanding the accessible Fourier frequency spectrum.
      2. Entangling Layer:
         Circular / Ladder CNOT entanglement connecting adjacent qubits.
      3. Variational Ansatz:
         General SU(2) Euler rotations Rot(alpha, beta, gamma) = Rz(alpha) Ry(beta) Rz(gamma)
         on every qubit at each layer.
      4. Measurement of Pauli Observables:
         Measures expectation values <Z_i> = <psi| sigma_z^(i) |psi> in [-1, +1] for each action.
      5. Softmax Policy Readout:
         Maps expectation values to action logits via learnable inverse temperature beta:
           logits = beta * <Z>
           pi(a|s) = Softmax(logits)
    """
    def __init__(self, in_features, n_qubits=9, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(JerbiDataReuploadingPQC, self).__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        adapter_hidden = 32 if n_qubits <= 4 else 64
        # Data Re-Uploading Adapters:
        # Maps input features s in R^{in_features} to rotation angles in [-pi, pi] for each layer and qubit.
        self.input_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, adapter_hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(adapter_hidden, n_qubits),
                nn.Tanh()
            )
            for _ in range(n_layers)
        ])

        # Trainable Variational Weights:
        # 3 Euler angles (alpha, beta, gamma) per qubit per layer: (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Trainable Softmax scaling factor beta (inverse temperature)
        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        else:
            self.register_buffer('output_scale', torch.tensor(float(output_scale)), persistent=False)

        # Quantum Device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(single_angles, weights):
            """
            single_angles: (L, n_qubits) rotation angles for one molecule
            weights: (L, n_qubits, 3) variational parameters
            """
            for l in range(n_layers):
                # 1. Data Re-Uploading: Inject classical state angles via Ry rotations
                for i in range(n_qubits):
                    qml.RY(single_angles[l, i], wires=i)

                # 2. Entanglement Layer: Circular CNOT ladder
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

                # 3. Variational SU(2) Rotations
                for i in range(n_qubits):
                    qml.Rot(
                        weights[l, i, 0],
                        weights[l, i, 1],
                        weights[l, i, 2],
                        wires=i
                    )

            # Measure Pauli-Z expectation values for each dedicated qubit / action
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def forward(self, state_features):
        """
        Args:
            state_features: Tensor of shape (B, in_features)
        Returns:
            logits: Tensor of shape (B, n_qubits)
            expvals: Tensor of shape (B, n_qubits) in [-1, 1]
        """
        device = state_features.device
        batch_size = state_features.size(0)

        # Compute data re-uploading angles for each layer: (B, L, n_qubits)
        angles_list = []
        for l in range(self.n_layers):
            angles_l = math.pi * self.input_adapters[l](state_features)  # scaled to [-pi, pi]
            angles_list.append(angles_l)
        angles = torch.stack(angles_list, dim=1)  # (B, L, n_qubits)

        angles_cpu = angles.cpu()
        weights_cpu = self.weights.cpu()

        expvals = []
        for b in range(batch_size):
            res = self.qnode(angles_cpu[b], weights_cpu)
            if isinstance(res, (list, tuple)):
                res = torch.stack(list(res))
            expvals.append(res)

        expvals = torch.stack(expvals, dim=0).to(device=device, dtype=state_features.dtype)
        if expvals.ndim == 1:
            expvals = expvals.unsqueeze(0)

        # Jerbi Softmax Readout: logits = beta * <Z>
        logits = self.output_scale * expvals
        return logits, expvals


# =============================================================================
# 2. DEDICATED ATOM QUANTUM POLICY HEAD (CNS 9-ELEMENT VOCABULARY & QM9 4-ATOM)
# =============================================================================

class JerbiAtomPolicy(nn.Module):
    """
    Jerbi Softmax-PQC for Heavy Atom Selection:
      - By default, 9 qubits correspond to the 9 heavy atom types in CNS vocabulary:
          Qubit 0: Carbon     (C)  - sp3/sp2/sp, tetrahedral/trigonal, valence 4
          Qubit 1: Nitrogen   (N)  - pyramidal/planar, valence 3
          Qubit 2: Oxygen     (O)  - bent, valence 2
          Qubit 3: Fluorine   (F)  - monovalent terminal halogen
          Qubit 4: Iodine     (I)  - monovalent heavy halogen
          Qubit 5: Phosphorus (P)  - tetrahedral, valence 3-5
          Qubit 6: Sulfur     (S)  - thioether/sulfonamide, valence 2
          Qubit 7: Chlorine   (Cl) - monovalent halogen
          Qubit 8: Bromine    (Br) - monovalent halogen
      - Alternatively, supports 4 qubits for QM9 heavy atoms {C, N, O, F}.
      - Multi-layer data re-uploading incorporates chemical compatibility signals.
      - Measures <Z_i> on each dedicated qubit.
      - Softmax action selection over candidate atom types.
    """
    def __init__(self, in_features=64, n_qubits=9, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(JerbiAtomPolicy, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Chemical compatibility adapter
        self.affinity_adapter = nn.Sequential(
            nn.Linear(in_features, 64 if n_qubits > 4 else 32),
            nn.LeakyReLU(0.2),
            nn.Linear(64 if n_qubits > 4 else 32, n_qubits),
            nn.Sigmoid()
        )

        # Jerbi Multi-Layer Data Re-Uploading PQC
        self.pqc = JerbiDataReuploadingPQC(
            in_features=in_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            output_scale=output_scale,
            learnable_scale=learnable_scale
        )

    def compute_atom_specific_compatibilities(
        self,
        bond_angle: torch.Tensor,
        free_valence: torch.Tensor,
        neighbor_type: torch.Tensor
    ) -> torch.Tensor:
        """Calculates physical & geometric chemical compatibility angles for candidate atoms."""
        B = bond_angle.size(0)
        device = bond_angle.device
        compat = torch.ones(B, self.n_qubits, device=device)

        # Qubit 0: Carbon — sp3/sp2 compatible
        diff_c = torch.abs(bond_angle - 1.911)
        compat[:, 0] = torch.clamp(1.0 - 0.4 * diff_c, 0.15, 1.0)

        # Qubit 1: Nitrogen — pyramidal
        diff_n = torch.abs(bond_angle - 1.866)
        base_n = torch.clamp(1.0 - 0.5 * diff_n, 0.1, 1.0)
        is_n_neighbor = (neighbor_type == 2).float()
        compat[:, 1] = base_n * (1.0 - 0.85 * is_n_neighbor)

        # Qubit 2: Oxygen — bent, anti-peroxide
        diff_o = torch.abs(bond_angle - 1.824)
        base_o = torch.clamp(1.0 - 0.6 * diff_o, 0.1, 1.0)
        is_o_neighbor = (neighbor_type == 3).float()
        is_strained = (bond_angle < 1.2).float()
        compat[:, 2] = base_o * (1.0 - 0.95 * is_o_neighbor) * (1.0 - 0.7 * is_strained)

        # Halogens (F, I, Cl, Br)
        is_c_neighbor = (neighbor_type == 1).float()
        is_terminal = (free_valence == 1).float()
        halogen_score = (0.1 + 0.9 * is_terminal) * is_c_neighbor

        compat[:, 3] = halogen_score   # F

        if self.n_qubits > 4:
            compat[:, 4] = halogen_score   # I
            # Qubit 5: Phosphorus
            diff_p = torch.abs(bond_angle - 1.745)
            compat[:, 5] = torch.clamp(1.0 - 0.5 * diff_p, 0.05, 0.7)
            # Qubit 6: Sulfur — bent
            diff_s = torch.abs(bond_angle - 1.824)
            base_s = torch.clamp(1.0 - 0.5 * diff_s, 0.1, 1.0)
            is_s_neighbor = (neighbor_type == 7).float()
            compat[:, 6] = base_s * (1.0 - 0.85 * is_s_neighbor)
            compat[:, 7] = halogen_score   # Cl
            compat[:, 8] = halogen_score   # Br

        return compat

    def forward(
        self,
        graph_emb: torch.Tensor,
        bond_angle: torch.Tensor = None,
        free_valence: torch.Tensor = None,
        neighbor_type: torch.Tensor = None
    ):
        B = graph_emb.size(0)
        device = graph_emb.device

        if bond_angle is None:
            bond_angle = torch.full((B,), 1.911, device=device)
        if free_valence is None:
            free_valence = torch.full((B,), 4.0, device=device)
        if neighbor_type is None:
            neighbor_type = torch.ones(B, dtype=torch.long, device=device)

        base_affinity = self.affinity_adapter(graph_emb)
        phys_compat = self.compute_atom_specific_compatibilities(
            bond_angle, free_valence, neighbor_type
        )
        atom_inputs = 0.5 * (base_affinity + phys_compat)

        # Jerbi Data Re-Uploading PQC
        logits, expvals = self.pqc(graph_emb)
        # Modulate with physical compatibility priors
        logits = logits + 0.5 * (atom_inputs - 0.5)

        probs = torch.softmax(logits, dim=-1)
        return logits, probs


# Dedicated Atom Quantum Policy alias for compatibility
DedicatedAtomQuantumPolicy = JerbiAtomPolicy


# =============================================================================
# 3. DEDICATED BOND QUANTUM POLICY HEAD (4 KEKULE BOND TYPES)
# =============================================================================

class JerbiEdgePolicy(nn.Module):
    """
    Jerbi Softmax-PQC for Bond Formation:
      - 4 dedicated qubits correspond to 4 Kekule bond types (Aromatic bond removed):
          Qubit 0: None (No bond)
          Qubit 1: Single bond (1.0)
          Qubit 2: Double bond (2.0)
          Qubit 3: Triple bond (3.0)
      - Multi-layer data re-uploading over concatenated pair features [h_u, h_v, graph_emb].
      - Measures <Z_0>, <Z_1>, <Z_2>, <Z_3>.
      - Softmax action selection over candidate bond types.
    """
    def __init__(self, in_features=192, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(JerbiEdgePolicy, self).__init__()
        self.n_qubits = 4
        self.n_layers = n_layers

        self.pqc = JerbiDataReuploadingPQC(
            in_features=in_features,
            n_qubits=4,
            n_layers=n_layers,
            output_scale=output_scale,
            learnable_scale=learnable_scale
        )

    def forward(self, pair_feat):
        logits, expvals = self.pqc(pair_feat)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


# Dedicated Bond Quantum Policy alias for compatibility
DedicatedBondQuantumPolicy = JerbiEdgePolicy


# =============================================================================
# 4. OPTIONAL 2-QUBIT BORN-RULE CIRCUIT & SIZE-MATCHED CLASSICAL BASELINE
# =============================================================================

class QuantumEdgePolicy2Q(nn.Module):
    """
    2-Qubit Variational Quantum Policy Head for Bond Formation (4 Basis States).
    Measures Born-rule computational basis probabilities:
      |00> <---> None (No Bond)     [Index 0]
      |01> <---> Single Bond (1.0)  [Index 1]
      |10> <---> Double Bond (2.0)  [Index 2]
      |11> <---> Triple Bond (3.0)  [Index 3]
    """
    def __init__(self, in_features=192, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(QuantumEdgePolicy2Q, self).__init__()
        self.n_qubits = 2
        self.n_layers = n_layers

        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        else:
            self.register_buffer('output_scale', torch.tensor(float(output_scale)), persistent=False)

        self.adapter = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.n_qubits),
            nn.Sigmoid()
        )

        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, self.n_qubits, 3))
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(single_input, weights):
            for i in range(self.n_qubits):
                qml.RY(single_input[i] * math.pi, wires=i)

            for layer in range(self.n_layers):
                qml.CNOT(wires=[0, 1])
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            return qml.probs(wires=[0, 1])

        self.qnode = circuit

    def forward(self, pair_feat):
        B = pair_feat.size(0)
        device = pair_feat.device
        bond_inputs = self.adapter(pair_feat)

        inputs_cpu = bond_inputs.cpu()
        weights_cpu = self.weights.cpu()

        probs_list = []
        for b in range(B):
            p = self.qnode(inputs_cpu[b], weights_cpu)
            if isinstance(p, (list, tuple)):
                p = torch.stack(list(p))
            probs_list.append(p)

        probs = torch.stack(probs_list, dim=0).to(device=device, dtype=torch.float32)
        probs = torch.clamp(probs, min=1e-6)
        logits = torch.log(probs) * self.output_scale
        return logits, probs


class ClassicalEdgeCircuit(nn.Module):
    """Size-matched classical control network for bond formation."""
    def __init__(self, in_features=192, hidden_dim=8, num_classes=4):
        super(ClassicalEdgeCircuit, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pair_feat):
        logits = self.net(pair_feat)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
