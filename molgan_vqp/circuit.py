import torch
import torch.nn as nn
import pennylane as qml
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


class QuantumPhysChemEvaluator(nn.Module):
    """
    Quantum Physicochemical Reward Circuit for MolGAN-VQP — CNS 9-Element Vocabulary.

    9 qubits, one per CNS physicochemical descriptor, matching the 9 real atom types
    (C, N, O, F, I, P, S, Cl, Br) in the Jerbi Dedicated Atom Quantum Policy Head:

      Qubit 0 : Molecular Weight        (MW)              calibrated to 100–500 Da CNS range
      Qubit 1 : Lipophilicity           (LogP)            calibrated to −1 to +5 CNS range
      Qubit 2 : H-Bond Donors           (HBD)             0–5 scale
      Qubit 3 : H-Bond Acceptors        (HBA)             0–10 scale
      Qubit 4 : Topological Polar SA    (TPSA)            0–90 Å² CNS target
      Qubit 5 : Rotatable Bonds         (RotBonds)        0–8 scale (CNS-MPO)
      Qubit 6 : Heteroatom Fraction     (HetAtomFrac)     0–0.5 scale
      Qubit 7 : Aromatic Ring Fraction  (AromaticFrac)    0–1 scale
      Qubit 8 : Ring Count              (RingCount)       0–5 scale

    Architecture:
      - Feature Encoding : RY(theta_i) per qubit
      - Data Re-Uploading : features re-encoded every layer
      - Ansatz Depth     : L = 3 layers of parameterized Rot(alpha, beta, gamma)
      - Entanglement     : Circular CNOT ring connecting all qubits
      - Readout          : Pauli-Z expectation on Qubit 0, normalized to [0, 1]
    """

    def __init__(self, n_qubits: int = 9, n_layers: int = 3):
        super(QuantumPhysChemEvaluator, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Variational weights: [n_layers, n_qubits, 3] for Rot(alpha, beta, gamma)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        def _circuit(features, weights):
            for l in range(self.n_layers):
                # 1. Feature encoding with RY rotations (data re-uploading)
                for i in range(self.n_qubits):
                    qml.RY(features[i], wires=i)

                # 2. Parameterized Euler rotations Rot(alpha, beta, gamma)
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

                # 3. Circular CNOT entangling ring over all qubits
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(_circuit, self.dev, interface="torch", diff_method="backprop")

    def extract_physchem_features(self, mol) -> torch.Tensor:
        """
        Extracts and scales physicochemical descriptors to rotation angles in [0, pi]
        calibrated to the CNS/drug chemical domain.
        Returns: Tensor of shape [n_qubits]
        """
        if mol is None:
            return torch.zeros(self.n_qubits, device=self.weights.device)

        pi = 3.141592653589793

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        if self.n_qubits == 4:
            # Scaled to 4-descriptor space
            theta_mw = pi * max(0.0, min((mw - 40.0) / 110.0, 1.0))
            theta_logp = pi * max(0.0, min((logp + 2.0) / 5.0, 1.0))
            theta_hbd = pi * max(0.0, min(hbd / 4.0, 1.0))
            theta_hba = pi * max(0.0, min(hba / 5.0, 1.0))
            return torch.tensor([theta_mw, theta_logp, theta_hbd, theta_hba],
                                dtype=torch.float32, device=self.weights.device)

        # 9-qubit full CNS scale
        tpsa = Descriptors.TPSA(mol)
        rotb = Lipinski.NumRotatableBonds(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        n_het = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (6, 1))
        ri = mol.GetRingInfo()
        n_rings = ri.NumRings()
        n_arom = sum(1 for r in ri.AtomRings()
                     if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in r))

        het_frac = n_het / max(n_heavy, 1)
        arom_frac = n_arom / max(n_rings, 1) if n_rings > 0 else 0.0

        theta_mw = pi * max(0.0, min((mw - 100.0) / 400.0, 1.0))
        theta_logp = pi * max(0.0, min((logp + 1.0) / 6.0, 1.0))
        theta_hbd = pi * max(0.0, min(hbd / 5.0, 1.0))
        theta_hba = pi * max(0.0, min(hba / 10.0, 1.0))
        theta_tpsa = pi * max(0.0, min(tpsa / 90.0, 1.0))
        theta_rotb = pi * max(0.0, min(rotb / 8.0, 1.0))
        theta_het = pi * max(0.0, min(het_frac / 0.5, 1.0))
        theta_arom = pi * max(0.0, min(arom_frac, 1.0))
        theta_rings = pi * max(0.0, min(n_rings / 5.0, 1.0))

        return torch.tensor(
            [theta_mw, theta_logp, theta_hbd, theta_hba,
             theta_tpsa, theta_rotb, theta_het, theta_arom, theta_rings],
            dtype=torch.float32,
            device=self.weights.device
        )

    def forward_single(self, features: torch.Tensor) -> torch.Tensor:
        """
        Executes quantum circuit for a single feature vector.
        Normalizes Pauli-Z expectation from [-1, 1] to [0, 1].
        """
        raw_val = self.qnode(features, self.weights)
        norm_val = (raw_val + 1.0) / 2.0
        return torch.clamp(norm_val, 0.0, 1.0).float()

    def evaluate_mols(self, mols) -> torch.Tensor:
        """
        Batch evaluation of a list of RDKit molecules.
        """
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(torch.tensor([0.0], dtype=torch.float32, device=self.weights.device))
                continue

            num_heavy = mol.GetNumHeavyAtoms()
            num_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

            min_heavy = 3 if self.n_qubits == 4 else 5
            min_c = 1 if self.n_qubits == 4 else 2
            if num_heavy < min_heavy or num_c < min_c:
                scores.append(torch.tensor([0.0], dtype=torch.float32, device=self.weights.device))
                continue

            feats = self.extract_physchem_features(mol)
            r_q = self.forward_single(feats)
            scores.append(r_q.float().view(1))

        return torch.stack(scores).float().view(-1, 1)
