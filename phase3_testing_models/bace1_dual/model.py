import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBBB_BACE1_Classifier(nn.Module):
    """
    Multi-Task Deep Neural Network for:
      Task 1: Blood-Brain Barrier (BBB) Permeability
      Task 2: BACE-1 (beta-Secretase 1) Enzyme Inhibition
    Input: 2058-D vector (2048 Morgan FP bits + 10 normalized physicochemical descriptors)
    """
    def __init__(self, in_features=2058, trunk_hidden=[512, 256], head_hidden=64, dropout_rate=0.3):
        super(DualBBB_BACE1_Classifier, self).__init__()

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_features, trunk_hidden[0]),
            nn.BatchNorm1d(trunk_hidden[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(trunk_hidden[0], trunk_hidden[1]),
            nn.BatchNorm1d(trunk_hidden[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        # Task 1: Blood-Brain Barrier (BBB) Head
        self.bbb_head = nn.Sequential(
            nn.Linear(trunk_hidden[1], head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_hidden, 1)
        )

        # Task 2: BACE-1 Inhibition Head
        self.bace1_head = nn.Sequential(
            nn.Linear(trunk_hidden[1], head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        feat = self.trunk(x)
        logit_bbb = self.bbb_head(feat)
        logit_bace1 = self.bace1_head(feat)
        return logit_bbb, logit_bace1

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            feat = self.trunk(x)
            prob_bbb = torch.sigmoid(self.bbb_head(feat))
            prob_bace1 = torch.sigmoid(self.bace1_head(feat))
        return prob_bbb, prob_bace1
