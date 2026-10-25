import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Ensure local imports work
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from features import extract_fingerprint, extract_physicochemical_descriptors
from model import DualBBB_BACE1_Classifier

DATA_PATH = os.path.join(CUR_DIR, 'master_bbb_bace1_dataset.csv')


class MultiTaskDataset(Dataset):
    def __init__(self, X, y_bbb, mask_bbb, y_bace1, mask_bace1):
        self.X = torch.from_numpy(X).float()
        self.y_bbb = torch.from_numpy(y_bbb).float().unsqueeze(-1)
        self.mask_bbb = torch.from_numpy(mask_bbb).float().unsqueeze(-1)
        self.y_bace1 = torch.from_numpy(y_bace1).float().unsqueeze(-1)
        self.mask_bace1 = torch.from_numpy(mask_bace1).float().unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y_bbb[idx], self.mask_bbb[idx],
                self.y_bace1[idx], self.mask_bace1[idx])


def main():
    print("=== Phase 3: Training Dual Blood-Brain Barrier & BACE-1 Classifier ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {len(df)} compounds")

    # Feature extraction
    print("Extracting 2048-bit Morgan Fingerprints + 10 physicochemical descriptors...")
    fps = []
    descs = []
    y_bbb_list = []
    mask_bbb_list = []
    y_bace1_list = []
    mask_bace1_list = []

    for _, row in df.iterrows():
        smi = row['smiles']
        fp = extract_fingerprint(smi)
        desc = extract_physicochemical_descriptors(smi)
        fps.append(fp)
        descs.append(desc)

        # BBB
        if pd.notna(row['bbb_label']):
            y_bbb_list.append(float(row['bbb_label']))
            mask_bbb_list.append(1.0)
        else:
            y_bbb_list.append(0.0)
            mask_bbb_list.append(0.0)

        # BACE-1
        if pd.notna(row['bace1_label']):
            y_bace1_list.append(float(row['bace1_label']))
            mask_bace1_list.append(1.0)
        else:
            y_bace1_list.append(0.0)
            mask_bace1_list.append(0.0)

    fps = np.array(fps, dtype=np.float32)
    descs = np.array(descs, dtype=np.float32)

    # Scale descriptors
    scaler = StandardScaler()
    scaled_descs = scaler.fit_transform(descs).astype(np.float32)
    with open(os.path.join(CUR_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved feature scaler to scaler.pkl")

    X = np.concatenate([fps, scaled_descs], axis=1).astype(np.float32)
    y_bbb = np.array(y_bbb_list, dtype=np.float32)
    mask_bbb = np.array(mask_bbb_list, dtype=np.float32)
    y_bace1 = np.array(y_bace1_list, dtype=np.float32)
    mask_bace1 = np.array(mask_bace1_list, dtype=np.float32)

    # Train / Test split (80 / 20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = MultiTaskDataset(X[train_idx], y_bbb[train_idx], mask_bbb[train_idx], y_bace1[train_idx], mask_bace1[train_idx])
    test_ds = MultiTaskDataset(X[test_idx], y_bbb[test_idx], mask_bbb[test_idx], y_bace1[test_idx], mask_bace1[test_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = DualBBB_BACE1_Classifier(in_features=2058).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_combined_auc = 0.0
    model_save_path = os.path.join(CUR_DIR, 'bbb_bace1_model.pt')

    print("\nStarting Multi-Task Training (50 Epochs)...")
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y_bbb, batch_m_bbb, batch_y_bace1, batch_m_bace1 in train_loader:
            batch_x = batch_x.to(device)
            batch_y_bbb = batch_y_bbb.to(device)
            batch_m_bbb = batch_m_bbb.to(device)
            batch_y_bace1 = batch_y_bace1.to(device)
            batch_m_bace1 = batch_m_bace1.to(device)

            logit_bbb, logit_bace1 = model(batch_x)

            # Masked losses
            loss_bbb = criterion(logit_bbb, batch_y_bbb) * batch_m_bbb
            loss_bace1 = criterion(logit_bace1, batch_y_bace1) * batch_m_bace1

            denom_bbb = batch_m_bbb.sum() + 1e-6
            denom_bace1 = batch_m_bace1.sum() + 1e-6

            loss = (loss_bbb.sum() / denom_bbb) + (loss_bace1.sum() / denom_bace1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Evaluate on Test Set every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 49:
            model.eval()
            p_bbb_all, y_bbb_all, m_bbb_all = [], [], []
            p_bace1_all, y_bace1_all, m_bace1_all = [], [], []

            with torch.no_grad():
                for bx, by_bbb, bm_bbb, by_bace1, bm_bace1 in test_loader:
                    bx = bx.to(device)
                    prob_bbb, prob_bace1 = model.predict_proba(bx)

                    p_bbb_all.append(prob_bbb.cpu().numpy())
                    y_bbb_all.append(by_bbb.numpy())
                    m_bbb_all.append(bm_bbb.numpy())

                    p_bace1_all.append(prob_bace1.cpu().numpy())
                    y_bace1_all.append(by_bace1.numpy())
                    m_bace1_all.append(bm_bace1.numpy())

            p_bbb_all = np.concatenate(p_bbb_all)
            y_bbb_all = np.concatenate(y_bbb_all)
            m_bbb_all = np.concatenate(m_bbb_all)

            p_bace1_all = np.concatenate(p_bace1_all)
            y_bace1_all = np.concatenate(y_bace1_all)
            m_bace1_all = np.concatenate(m_bace1_all)

            # BBB metrics
            valid_bbb = (m_bbb_all.flatten() == 1.0)
            auc_bbb = roc_auc_score(y_bbb_all[valid_bbb], p_bbb_all[valid_bbb])
            acc_bbb = accuracy_score(y_bbb_all[valid_bbb], (p_bbb_all[valid_bbb] >= 0.5).astype(int))

            # BACE-1 metrics
            valid_bace1 = (m_bace1_all.flatten() == 1.0)
            auc_bace1 = roc_auc_score(y_bace1_all[valid_bace1], p_bace1_all[valid_bace1])
            acc_bace1 = accuracy_score(y_bace1_all[valid_bace1], (p_bace1_all[valid_bace1] >= 0.5).astype(int))

            comb_auc = (auc_bbb + auc_bace1) / 2.0
            print(f"Epoch [{epoch+1:02d}/50] | Loss: {total_loss/n_batches:.4f} | "
                  f"BBB AUC: {auc_bbb:.4f} (Acc: {acc_bbb*100:.1f}%) | "
                  f"BACE-1 AUC: {auc_bace1:.4f} (Acc: {acc_bace1*100:.1f}%)")

            if comb_auc > best_combined_auc:
                best_combined_auc = comb_auc
                torch.save(model.state_dict(), model_save_path)

    print(f"\nTraining Complete! Best Combined ROC-AUC: {best_combined_auc:.4f}")
    print(f"Saved weights to: {model_save_path}")


if __name__ == '__main__':
    main()
