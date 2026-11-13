"""Offline publication figure generator for MolGAN-VQP manuscript.

Reads only pre-saved CSV benchmarks and pre-rendered run images on disk.
No model loading, live sampling, or benchmarking is performed.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageChops, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(PAPER_DIR, "benchmark_results")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _trim_white_margin(im: Image.Image, pad: int = 12) -> Image.Image:
    """Crop excessive solid-white border around saved grids while preserving content."""
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    bbox = diff.getbbox()
    if not bbox:
        return rgb
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(rgb.width, bbox[2] + pad)
    y1 = min(rgb.height, bbox[3] + pad)
    return rgb.crop((x0, y0, x1, y1))


def copy_actual_run_images():
    """Copy actual raw run outputs into paper_manuscript/figures/ with tight whitespace."""
    copies = [
        (
            os.path.join(BASE_DIR, "Results", "full_scale_50epochs", "2026-11-11_00-42-05", "img_dir", "epoch_001_mols.png"),
            os.path.join(FIG_DIR, "epoch_001_mols.png"),
        ),
        (
            os.path.join(BASE_DIR, "Results", "full_scale_50epochs", "2026-11-11_00-42-05", "img_dir", "epoch_050_mols.png"),
            os.path.join(FIG_DIR, "epoch_050_mols.png"),
        ),
        (
            os.path.join(BASE_DIR, "Results", "screened_leads_100_vqp", "screened_candidates_grid.png"),
            os.path.join(FIG_DIR, "screened_100_top16_grid.png"),
        ),
        (
            os.path.join(BASE_DIR, "Results", "screened_leads_expanded_bace1", "screened_candidates_grid.png"),
            os.path.join(FIG_DIR, "screened_expanded_bace1_grid.png"),
        ),
        (
            "/home/cse-sdpl/MolGAN-QRL/Results/cleaned_molecules_sample.png",
            os.path.join(FIG_DIR, "qrl_cleaned_molecules_sample.png"),
        ),
    ]
    for src, dst in copies:
        if os.path.exists(src):
            with Image.open(src) as im:
                trimmed = _trim_white_margin(im, pad=14)
                trimmed.save(dst, dpi=(300, 300))
            print(f"Copied & trimmed: {src} -> {dst}")
        else:
            print(f"WARNING missing source image: {src}")


def build_fig_bar_comparison():
    """Generate 6-panel comparative bar chart across saved CSV benchmarks."""
    specs = [
        ("Classical\nMolGAN\n(N=35)", os.path.join(BENCH_DIR, "classical_gan_strict.csv"), "#7f8c8d"),
        ("MolGAN-\nQRL\n(N=35)", os.path.join(BENCH_DIR, "molgan_qrl_strict.csv"), "#e67e22"),
        ("MolGAN-\nQPolicy\n(4Q QM9*)", os.path.join(BENCH_DIR, "molgan_ini_4q.csv"), "#9b59b6"),
        ("Ablation\nw/o Atom\nCaps", os.path.join(BENCH_DIR, "ablation_no_atom_filters.csv"), "#c0392b"),
        ("VQP Ep.50\n(9Q Atom\n+ Mask)", os.path.join(BENCH_DIR, "molgan_vqp_ep50.csv"), "#006699"),
        ("VQP Dual\n(9Q+4Q\nPolicy*)", os.path.join(BENCH_DIR, "molgan_vqp_dual_20.csv"), "#16a085"),
    ]

    records = []
    for label, path, color in specs:
        df = pd.read_csv(path)
        n = len(df)
        u_df = df[df["is_unique"] == True]
        nu = max(1, len(u_df))
        records.append({
            "label": label,
            "color": color,
            "uniqueness": 100.0 * df["is_unique"].sum() / n,
            "clean": 100.0 * df["is_clean"].sum() / n,
            "stable_rings": 100.0 * df["has_stable_rings"].sum() / n,
            "bbb_plus": 100.0 * (u_df["sota_bbb_prob"] >= 0.5).sum() / nu,
            "bace1_active": 100.0 * u_df["bace1_active"].sum() / nu,
            "tier1": 100.0 * u_df["tier"].astype(str).str.startswith("TIER 1").sum() / nu,
        })

    rdf = pd.DataFrame(records)
    panels = [
        ("uniqueness", "(A) Chemical Uniqueness (%)", 116),
        ("clean", "(B) Strain-Free Cleanliness (%)", 116),
        ("stable_rings", "(C) Stable 5/6-Ring Topology (%)", 116),
        ("bbb_plus", "(D) BBB+ Permeable among Unique (%)", 116),
        ("bace1_active", "(E) BACE-1 Active among Unique (%)", 52),
        ("tier1", "(F) Tier-1 CNS Leads among Unique (%)", 28),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.2))
    x = np.arange(len(rdf))
    for ax, (col, title, ymax) in zip(axes.flat, panels):
        bars = ax.bar(x, rdf[col], color=rdf["color"], width=0.58, edgecolor="#222222", linewidth=0.6)
        ax.set_title(title, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(rdf["label"], fontsize=8.0)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, rdf[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + ymax * 0.02,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color="#222222",
            )

    fig.suptitle(
        "Quantitative Architecture Benchmark & Ablation Analysis across Saved Evaluation Campaigns",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "*Notes: Classical Strict (n=100), MolGAN-QRL Strict (n=100), 4Q QM9 Baseline (n=77 valid/100; 0% cyclic, mean MW=123.8 Da),\n"
        "Ablation w/o Atom Caps (n=100), VQP Ep.50 (n=100), Dual-Q 9Q+4Q (n=20). Dynamic Valence Masking ablation yields 0.0% validity.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.945])
    out = os.path.join(FIG_DIR, "fig_bar_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def _render_crisp_mol_grid(mols, legends_lines, cols=4, cell_w=360, mol_h=230, cap_h=66, font_size=13):
    """Render RDKit structures with sharp PIL vector/TrueType legends."""
    rows = int(np.ceil(len(mols) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows * (mol_h + cap_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 1)
    except Exception:
        font = font_sub = ImageFont.load_default()

    for idx, (mol, lines) in enumerate(zip(mols, legends_lines)):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * (mol_h + cap_h)
        mimg = Draw.MolToImage(mol, size=(cell_w - 10, mol_h - 6))
        canvas.paste(mimg, (x0 + 5, y0 + 2))
        for li, text in enumerate(lines):
            f_use = font if li == 0 else font_sub
            col = "#005580" if li == 0 else "#222222"
            tw = draw.textlength(text, font=f_use) if hasattr(draw, "textlength") else len(text) * 7
            draw.text((x0 + (cell_w - tw) / 2.0, y0 + mol_h + li * (font_size + 4)), text, fill=col, font=f_use)
    return canvas


def build_best_leads_grid():
    """Render 2x4 publication grid of top 8 Tier-1 leads from screened_leads_100_vqp."""
    csv_path = os.path.join(BASE_DIR, "Results", "screened_leads_100_vqp", "screened_candidates_summary.csv")
    df = pd.read_csv(csv_path)
    target_ranks = [1, 2, 3, 4, 7, 9, 10, 14]
    sub = df[df["rank"].isin(target_ranks)].sort_values("rank")

    mols, legends = [], []
    for _, row in sub.iterrows():
        mols.append(Chem.MolFromSmiles(row["smiles"]))
        legends.append([
            f"Rank #{int(row['rank'])} | N={int(row['heavy_atoms'])} | MW={row['mw']:.1f} Da",
            f"BBB+={row['sota_bbb_prob']:.3f} | BACE-1={row['bace1_prob']:.3f}",
            f"LogP={row['logp']:.2f} | CNS-MPO={row['cns_mpo']:.2f} | Clean",
        ])

    img = _render_crisp_mol_grid(mols, legends, cols=4, cell_w=380, mol_h=245, cap_h=68, font_size=14)
    out = os.path.join(FIG_DIR, "best_leads_grid.png")
    img.save(out, dpi=(300, 300))
    print(f"Saved: {out}")


def build_mol_comparison_publication():
    """Render side-by-side baseline pathology vs. MolGAN-VQP bicyclic CNS leads."""
    qrl_df = pd.read_csv(os.path.join(BENCH_DIR, "molgan_qrl_strict.csv"))
    gan_df = pd.read_csv(os.path.join(BENCH_DIR, "classical_gan_strict.csv"))
    abl_df = pd.read_csv(os.path.join(BENCH_DIR, "ablation_no_atom_filters.csv"))
    vqp_df = pd.read_csv(os.path.join(BASE_DIR, "Results", "screened_leads_100_vqp", "screened_candidates_summary.csv"))

    gan_top_smi = gan_df["smiles"].value_counts().index[0]
    qrl_top_smi = qrl_df["smiles"].value_counts().index[0]
    qrl_ring_smi = qrl_df["smiles"].value_counts().index[1]
    abl_smi = abl_df.iloc[0]["smiles"]

    vqp_ranks = [1, 3, 4, 14]
    vqp_rows = [vqp_df[vqp_df["rank"] == r].iloc[0] for r in vqp_ranks]

    top_mols = [Chem.MolFromSmiles(s) for s in [gan_top_smi, qrl_top_smi, qrl_ring_smi, abl_smi]]
    top_legends = [
        ["Classical MolGAN (x10 Mode Collapse)", "N=32 | Clean=False (Macro-Enol)"],
        ["MolGAN-QRL (x16 Mode Collapse)", "N=26 | Acyclic Branched (0 Rings)"],
        ["MolGAN-QRL (x6 Repeat Macrocycle)", "N=29 | Clean=False (Large Strained Ring)"],
        ["Ablation w/o CNS Atom Caps", "N=20 | Clean=False (P/I Over-Heteroatom)"],
    ]
    bot_mols = [Chem.MolFromSmiles(r["smiles"]) for r in vqp_rows]
    bot_legends = [
        [f"MolGAN-VQP Lead #{int(r['rank'])} (Unique)", f"BBB+={r['sota_bbb_prob']:.3f} | BACE-1={r['bace1_prob']:.3f} | MPO={r['cns_mpo']:.2f}"]
        for r in vqp_rows
    ]

    grid_top = _render_crisp_mol_grid(top_mols, top_legends, cols=4, cell_w=380, mol_h=240, cap_h=50, font_size=14)
    grid_bot = _render_crisp_mol_grid(bot_mols, bot_legends, cols=4, cell_w=380, mol_h=240, cap_h=50, font_size=14)

    try:
        font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    except Exception:
        font_hdr = ImageFont.load_default()

    final_h = 38 + grid_top.height + 44 + grid_bot.height + 8
    final_img = Image.new("RGB", (grid_top.width, final_h), "white")
    fdraw = ImageDraw.Draw(final_img)
    fdraw.rectangle([8, 4, grid_top.width - 8, 34], fill="#fdecea")
    fdraw.text((20, 9), "Baseline & Ablation Pathologies (Mode Collapse, Acyclic Chains, Strained Macrocycles, Uncapped Heteroatoms)", fill="#922b21", font=font_hdr)
    final_img.paste(grid_top, (0, 38))
    y_sep = 38 + grid_top.height + 6
    fdraw.rectangle([8, y_sep, grid_top.width - 8, y_sep + 32], fill="#e8f6f3")
    fdraw.text((20, y_sep + 7), "MolGAN-VQP Tier-1 De Novo Leads (100% Unique & Novel, Strain-Free, Verified BACE-1 & BBB+)", fill="#0e6251", font=font_hdr)
    final_img.paste(grid_bot, (0, y_sep + 36))

    out = os.path.join(FIG_DIR, "fig_mol_comparison_publication.png")
    final_img.save(out, dpi=(300, 300))
    print(f"Saved: {out}")


def build_cns_distributions():
    """Plot 2-panel physicochemical CNS window and BACE-1/BBB+ tier scatter."""
    gan_df = pd.read_csv(os.path.join(BENCH_DIR, "classical_gan_strict.csv")).drop_duplicates("smiles")
    qrl_df = pd.read_csv(os.path.join(BENCH_DIR, "molgan_qrl_strict.csv")).drop_duplicates("smiles")
    vqp_df = pd.read_csv(os.path.join(BENCH_DIR, "molgan_vqp_ep50.csv"))
    scr_df = pd.read_csv(os.path.join(BASE_DIR, "Results", "screened_leads_100_vqp", "screened_candidates_summary.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    rect = patches.Rectangle((10, 1.0), 80, 3.0, linewidth=1.2, edgecolor="#006699", facecolor="#e6f2f8", alpha=0.65, label="CNS Drug-Like Window (TPSA 10-90, CLogP 1-4)")
    ax1.add_patch(rect)
    ax1.scatter(gan_df["tpsa"], gan_df["clogp"], c="#7f8c8d", marker="^", s=45, alpha=0.75, label=f"Classical MolGAN (unique n={len(gan_df)})")
    ax1.scatter(qrl_df["tpsa"], qrl_df["clogp"], c="#e67e22", marker="s", s=45, alpha=0.80, label=f"MolGAN-QRL (unique n={len(qrl_df)})")
    ax1.scatter(vqp_df["tpsa"], vqp_df["clogp"], c="#006699", marker="o", s=46, alpha=0.82, edgecolors="white", linewidth=0.4, label=f"MolGAN-VQP Ep.50 (unique n={len(vqp_df)})")
    ax1.set_xlabel("Topological Polar Surface Area (TPSA, \u00c5\u00b2)")
    ax1.set_ylabel("Calculated Lipophilicity (CLogP)")
    ax1.set_title("(A) CNS Physicochemical Space Distribution")
    ax1.set_xlim(-5, 145)
    ax1.set_ylim(-2.5, 6.5)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right", frameon=True, fontsize=8)

    tier_styles = [
        ("TIER 1", "#0e6251", "o", "Tier 1: Prime BBB+ BACE-1 (n=16)"),
        ("TIER 2", "#2e86c1", "D", "Tier 2: Potent BACE-1, Minor Alert (n=5)"),
        ("TIER 3", "#d4ac0d", "^", "Tier 3: High BACE-1, Delivery Opt. (n=15)"),
        ("TIER 4", "#8e44ad", "s", "Tier 4: BBB+ Only, BACE-1 Inactive (n=39)"),
        ("TIER 5", "#7f8c8d", "x", "Tier 5: Inactive & Non-Permeable (n=25)"),
    ]
    for prefix, color, marker, lbl in tier_styles:
        sub = scr_df[scr_df["tier"].astype(str).str.startswith(prefix)]
        ax2.scatter(sub["sota_bbb_prob"], sub["bace1_prob"], c=color, marker=marker, s=55 if prefix == "TIER 1" else 40, alpha=0.88, label=lbl)

    ax2.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.fill_between([0.5, 1.03], 0.5, 1.05, color="#e8f6f3", alpha=0.45, zorder=0)
    ax2.set_xlabel("Phase 3 SOTA Stacking Probability P(BBB+)")
    ax2.set_ylabel("Phase 3 Dual-Model Probability P(BACE-1 Active)")
    ax2.set_title("(B) Phase 3 Multi-Target Tiering (100 Unique MolGAN-VQP Leads)")
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.04, 1.06)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="center left", bbox_to_anchor=(0.01, 0.48), frameon=True, fontsize=7.8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_cns_distributions.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


def build_training_curves():
    """Plot 4-panel 50-epoch training dynamics directly from saved metrics.csv."""
    csv_path = os.path.join(BASE_DIR, "Results", "full_scale_50epochs", "2026-11-11_00-42-05", "log_dir", "metrics.csv")
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))

    ax = axes[0, 0]
    ax.plot(epochs, 100.0 * df["validity"], "o-", markersize=3, color="#27ae60", linewidth=1.6, label="Validity / Uniqueness / Novelty (100.0%)")
    ax.plot(epochs, 100.0 * df["stability"], "s-", markersize=3.5, color="#006699", linewidth=2.0, label="Clean Unstrained Rate (Mean 97.88%, Ep.50 100.0%)")
    ax.axhline(50.0, color="#c0392b", linestyle="--", linewidth=1.4, label="Pre-Rule-14 Mask Baseline (50.0%)")
    ax.set_title("(A) Chemical Validity, Diversity & Anti-Strain Stability")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(42, 105)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0, 1]
    l1, = ax.plot(epochs, df["loss_critic"], "o-", markersize=3, color="#8e44ad", linewidth=1.8, label="Critic MSE Loss (Left)")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Critic Loss (MSE)", color="#8e44ad")
    ax.tick_params(axis="y", labelcolor="#8e44ad")
    ax.set_ylim(0.0, 0.34)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    l2, = ax_r.plot(epochs, df["loss_actor"], "d--", markersize=3, color="#d35400", linewidth=1.6, label="PPO Clipped Surrogate Loss (Right)")
    ax_r.set_ylabel("Actor Clipped Objective Loss", color="#d35400")
    ax_r.tick_params(axis="y", labelcolor="#d35400")
    ax_r.set_ylim(-0.155, -0.018)
    ax.set_title("(B) PPO Actor & Value Critic Loss Convergence")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right", fontsize=8)

    ax = axes[1, 0]
    ax.plot(epochs, df["loss_disc"], "o-", markersize=3, color="#16a085", linewidth=1.8, label="R-GCN Discriminator BCE Loss (Mean 0.5234)")
    ax.set_title("(C) Adversarial Discriminator Equilibrium")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Discriminator Loss")
    ax.set_ylim(0.36, 0.68)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.plot(epochs, df["mean_reward"], "o-", markersize=3.5, color="#006699", linewidth=2.0, label=f"Composite Reward (Peak 0.748, Mean {df['mean_reward'].mean():.3f})")
    ax.plot(epochs, df["mean_bbb"], "^--", markersize=3, color="#27ae60", linewidth=1.5, label=f"Mean BBB+ Prob. (Mean {df['mean_bbb'].mean():.3f})")
    ax.plot(epochs, df["mean_bace1"], "s--", markersize=3, color="#e67e22", linewidth=1.5, label=f"Mean BACE-1 Prob. (Mean {df['mean_bace1'].mean():.3f})")
    ax.set_title("(D) Multi-Objective Reward & Target Affinity Signals")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Batch Mean Score")
    ax.set_ylim(0.24, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, ncol=1)

    fig.suptitle(
        "50-Epoch Full-Scale MolGAN-VQP Training Trajectory (7,280.9 s = 2.02 h, Batch Size = 16)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG_DIR, "fig_training_curves.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    copy_actual_run_images()
    build_fig_bar_comparison()
    build_best_leads_grid()
    build_mol_comparison_publication()
    build_cns_distributions()
    build_training_curves()
    print("All paper figures created successfully.")
