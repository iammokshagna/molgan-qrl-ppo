import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

try:
    from phase3_bbb_bace1.features import extract_combined_features
    from phase3_bbb_bace1.model import DualBBB_BACE1_Classifier
    from phase3_bbb_bace1.justification import generate_bace1_bbb_justification
except ImportError:
    from features import extract_combined_features
    from model import DualBBB_BACE1_Classifier
    from justification import generate_bace1_bbb_justification

MODEL_PATH = os.path.join(CUR_DIR, 'bbb_bace1_model.pt')
SCALER_PATH = os.path.join(CUR_DIR, 'scaler.pkl')

_CACHED_MODEL = None
_CACHED_SCALER = None


def get_classifier_and_scaler(device=None):
    global _CACHED_MODEL, _CACHED_SCALER
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _CACHED_MODEL is None or _CACHED_SCALER is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Model or scaler not found in {CUR_DIR}. Please run train_bace1_bbb_model.py first.")

        with open(SCALER_PATH, 'rb') as f:
            _CACHED_SCALER = pickle.load(f)

        model = DualBBB_BACE1_Classifier(in_features=2058).to(device)
        weights = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(weights)
        model.eval()
        _CACHED_MODEL = model

    return _CACHED_MODEL, _CACHED_SCALER, device


def evaluate_molecule(smiles, device=None):
    """
    Evaluates a single SMILES strictly for:
      1. Blood-Brain Barrier (BBB) Permeability
      2. BACE-1 (beta-Secretase 1) Enzyme Inhibition
      3. Scientific justification and actionable tier
    """
    model, scaler, dev = get_classifier_and_scaler(device)
    feats = extract_combined_features(smiles, scaler=scaler)
    feats_t = torch.from_numpy(feats).unsqueeze(0).to(dev)

    prob_bbb, prob_bace1 = model.predict_proba(feats_t)
    p_bbb = float(prob_bbb.cpu().item())
    p_bace1 = float(prob_bace1.cpu().item())

    report = generate_bace1_bbb_justification(smiles, p_bbb, p_bace1)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMILES strictly for BBB Permeability and BACE-1 Inhibition")
    parser.add_argument('--smiles', type=str, default=None, help='Single SMILES string')
    parser.add_argument('--input_csv', type=str, default=None, help='Input CSV with a "smiles" column')
    parser.add_argument('--output_json', type=str, default=None, help='Output JSON file path')
    args = parser.parse_args()

    if args.smiles:
        report = evaluate_molecule(args.smiles)
        print(json.dumps(report, indent=2))
        return

    if args.input_csv:
        import pandas as pd
        df = pd.read_csv(args.input_csv)
        col = 'smiles' if 'smiles' in df.columns else ('SMILES' if 'SMILES' in df.columns else df.columns[0])
        print(f"Evaluating {len(df)} compounds from {args.input_csv}...")
        results = []
        for s in df[col]:
            rep = evaluate_molecule(s)
            results.append(rep)
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved reports to {args.output_json}")


if __name__ == '__main__':
    main()
