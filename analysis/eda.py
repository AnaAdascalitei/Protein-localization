"""Analiza exploratorie: cum arată datele înainte de orice model.

Rezultate în results/eda/:
    class_counts.png        câte proteine are fiecare clasă
    features_by_class.png   distribuția câtorva features-cheie pe clase
    summary_by_class.csv    media fiecărui feature pe fiecare clasă

Rulare:
    python analysis/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["Nucleus", "Cytoplasm", "Membrane", "Extracellular"]
KEY = {
    "tm_segments": "segmente transmembranare",
    "nterm_max_hydro_window": "hidrofobicitate max. la capătul N",
    "max_basic_window8": "max. K/R în 8 aminoacizi",
    "aa_C": "fracția de cisteină (C)",
    "gravy": "GRAVY (hidrofobicitate medie)",
    "isoelectric_point": "punct izoelectric (pI)",
}


def main(data_path: Path = ROOT / "data" / "processed" / "dataset.csv", out: Path = ROOT / "results" / "eda"):
    df = pd.read_csv(data_path)
    out.mkdir(parents=True, exist_ok=True)
    order = [c for c in ORDER if c in set(df["label"])]

    counts = df["label"].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(counts.index, counts.values, color="#378ADD")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("proteine")
    ax.set_title("Proteine pe clasă")
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title) in zip(axes.flat, KEY.items()):
        ax.boxplot([df.loc[df["label"] == c, col] for c in order], showfliers=False)
        ax.set_xticks(range(1, len(order) + 1), order, rotation=20)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / "features_by_class.png", dpi=130)
    plt.close(fig)

    num = df.drop(columns=["accession", "entry_name", "gene", "protein_name", "group"])
    num.groupby("label").mean().reindex(order).T.round(3).to_csv(out / "summary_by_class.csv")
    print(counts.to_string())
    print(f"Grafice salvate în {out}")


if __name__ == "__main__":
    main()
