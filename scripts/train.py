"""Antrenează și evaluează modelele.

Împărțire: 80% train / 20% test, pe GRUPURI de proteine asemănătoare
(o familie întreagă e ori în train, ori în test). Pe train facem validare
încrucișată în 5 părți (tot pe grupuri) ca să comparăm modelele; setul de test
e folosit o singură dată, la final.

Rezultate în results/:
    metrics.json                 macro-F1 și acuratețe (validare + test) pentru fiecare model
    report_<model>.txt           precizie / recall / F1 pe fiecare clasă
    confusion_<model>.png        matricea de confuzie
    importance.png               ce features contează cel mai mult (permutation importance)

Rulare:
    python scripts/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,  # noqa: E402
                             f1_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from localization.labels import CLASSES  # noqa: E402

META = ["accession", "entry_name", "gene", "protein_name", "label", "group"]
SEED = 42


def models() -> dict:
    return {
        "baseline_majoritar": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                            class_weight="balanced", random_state=SEED),
    }


def group_split(y, groups, test_fraction: float = 0.2):
    """O singură împărțire train/test, stratificată pe clase și separată pe grupuri."""
    cv = StratifiedGroupKFold(n_splits=round(1 / test_fraction), shuffle=True, random_state=SEED)
    return next(cv.split(np.zeros(len(y)), y, groups))


def main(data_path: Path = ROOT / "data" / "processed" / "dataset.csv", out: Path = ROOT / "results") -> dict:
    df = pd.read_csv(data_path)
    X, y, groups = df.drop(columns=META), df["label"], df["group"]
    tr, te = group_split(y, groups)
    assert not set(groups.iloc[tr]) & set(groups.iloc[te]), "grupuri comune între train și test"
    print(f"Train: {len(tr)} proteine | Test: {len(te)} proteine | {X.shape[1]} features")

    out.mkdir(parents=True, exist_ok=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    labels = [c for c in CLASSES if c in set(y)]
    for name, model in models().items():
        scores = cross_validate(model, X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr], cv=cv,
                                scoring=["f1_macro", "accuracy"], n_jobs=1)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        results[name] = {
            "cv_macro_f1": round(scores["test_f1_macro"].mean(), 3),
            "cv_macro_f1_std": round(scores["test_f1_macro"].std(), 3),
            "test_macro_f1": round(f1_score(y.iloc[te], pred, average="macro"), 3),
            "test_accuracy": round(accuracy_score(y.iloc[te], pred), 3),
        }
        r = results[name]
        print(f"{name:<20} CV macro-F1 {r['cv_macro_f1']:.3f} ± {r['cv_macro_f1_std']:.3f}"
              f" | test macro-F1 {r['test_macro_f1']:.3f} | test acuratețe {r['test_accuracy']:.3f}")
        (out / f"report_{name}.txt").write_text(
            classification_report(y.iloc[te], pred, labels=labels, zero_division=0), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(y.iloc[te], pred, labels=labels, ax=ax,
                                                colorbar=False, xticks_rotation=30)
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name}.png", dpi=130)
        plt.close(fig)

    # Ce features contează: amestecăm o coloană și vedem cât scade macro-F1 pe test
    best = max((n for n in results if n != "baseline_majoritar"), key=lambda n: results[n]["cv_macro_f1"])
    model = models()[best].fit(X.iloc[tr], y.iloc[tr])
    imp = permutation_importance(model, X.iloc[te], y.iloc[te], scoring="f1_macro",
                                 n_repeats=5, random_state=SEED, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(X.columns[order][::-1], imp.importances_mean[order][::-1], color="#1D9E75")
    ax.set_xlabel("scăderea macro-F1 când coloana e amestecată")
    ax.set_title(f"Cele mai importante 15 features ({best})")
    fig.tight_layout()
    fig.savefig(out / "importance.png", dpi=130)
    plt.close(fig)

    results["_best_model"] = best
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nCel mai bun model (după validare): {best}. Rezultate în {out}")
    return results


if __name__ == "__main__":
    main()
