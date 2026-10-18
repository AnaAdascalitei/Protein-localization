"""Din datele brute UniProt construiește setul de date final (o proteină = un rând).

Pași:
    1. citește cel mai recent fișier data/raw/uniprot_human_*.tsv
    2. eticheta: din textul de localizare (vezi localization/labels.py)
    3. filtre: secvențe de 50–5.000 de aminoacizi, fără duplicate exacte
    4. features: calculate din secvență (vezi localization/features.py)
    5. grupuri de similaritate pentru împărțirea train/test (localization/split.py)

Rezultat:
    data/processed/dataset.csv        setul de date (ID-uri, eticheta, grupul, features)
    data/processed/label_report.csv   câte proteine au fost păstrate / scoase și de ce

Rulare:
    python extraction/build_dataset.py
    python extraction/build_dataset.py --evidence any     # și locațiile deduse, nu doar cele experimentale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from localization.features import all_features, clean  # noqa: E402
from localization.labels import label  # noqa: E402
from localization.split import similarity_groups  # noqa: E402

MIN_LEN, MAX_LEN = 50, 5000


def latest_raw(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("uniprot_human_*.tsv"))
    if not files:
        raise SystemExit("Nu există date brute. Rulați întâi: python extraction/collect_uniprot.py")
    return files[-1]


def build(raw: pd.DataFrame, evidence: str = "experimental", threshold: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.rename(columns={
        "Entry": "accession", "Entry Name": "entry_name", "Protein names": "protein_name",
        "Gene Names (primary)": "gene", "Length": "length_uniprot", "Sequence": "sequence",
        "Subcellular location [CC]": "location_text", "Annotation": "annotation_score",
    })
    labels = raw["location_text"].apply(lambda t: label(t, evidence))
    raw["label"] = [l for l, _ in labels]
    raw["reason"] = [r for _, r in labels]

    seq = raw["sequence"].fillna("").map(clean)
    raw.loc[raw["label"].notna() & ((seq.str.len() < MIN_LEN) | (seq.str.len() > MAX_LEN)), "reason"] = "lungime"
    raw.loc[raw["reason"] == "lungime", "label"] = None
    dup = raw["label"].notna() & seq.duplicated(keep="first")
    raw.loc[dup, ["label", "reason"]] = [None, "duplicat"]

    report = raw["reason"].value_counts().rename_axis("motiv").reset_index(name="proteine")

    kept = raw[raw["label"].notna()].reset_index(drop=True)
    feats = pd.DataFrame([all_features(s) for s in kept["sequence"]])
    kept["group"] = similarity_groups(kept["sequence"].map(clean).tolist(), threshold)
    cols = ["accession", "entry_name", "gene", "protein_name", "label", "group"]
    return pd.concat([kept[cols], feats], axis=1), report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, help="fișierul brut (implicit: cel mai recent)")
    ap.add_argument("--evidence", choices=["experimental", "any"], default="experimental")
    ap.add_argument("--similarity", type=float, default=0.4, help="prag de similaritate pentru grupuri")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    raw_path = args.raw or latest_raw(ROOT / "data" / "raw")
    print(f"Citesc {raw_path.name}")
    raw = pd.read_csv(raw_path, sep="\t", dtype=str)
    data, report = build(raw, args.evidence, args.similarity)

    args.out.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out / "dataset.csv", index=False)
    report.to_csv(args.out / "label_report.csv", index=False)

    print(f"\nProteine în fișierul brut: {len(raw)}")
    print(report.to_string(index=False))
    print(f"\nSet final: {len(data)} proteine, {data.shape[1] - 6} features")
    print(data["label"].value_counts().to_string())
    sizes = data["group"].value_counts()
    print(f"Grupuri de similaritate: {len(sizes)} (cel mai mare: {sizes.iloc[0]} proteine)")
    if sizes.iloc[0] > 0.1 * len(data):
        print("  ATENȚIE: cel mai mare grup are peste 10% din date; încercați un prag mai mare, "
              "de ex. --similarity 0.5")
    print(f"Salvat în {args.out}")


if __name__ == "__main__":
    main()
