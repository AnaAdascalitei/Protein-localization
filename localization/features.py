"""Calculul features-urilor dintr-o secvență de aminoacizi.

Fiecare funcție primește secvența (șir de litere) și întoarce numere.
Explicațiile pe larg (ce e fiecare și de ce contează) sunt în documentul proiectului.

Grupe de features:
    1. de bază          lungime, masă
    2. compoziție       fracția fiecăruia din cei 20 de aminoacizi + grupe (hidrofobi, încărcați ...)
    3. fizico-chimice   pI, sarcină, GRAVY, aromaticitate, instabilitate, indice alifatic, structură secundară
    4. pe regiuni       N-terminal (peptid semnal), C-terminal (semnale de reținere),
                        segmente transmembranare, semnale de localizare nucleară (NLS)
"""
from __future__ import annotations

import math
import re

from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Scala Kyte–Doolittle (1982): cât de „hidrofob” (fuge de apă) e fiecare aminoacid.
# Pozitiv = hidrofob (I, V, L, F ...), negativ = hidrofil (R, K, D, E ...).
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

GROUPS = {
    "hydrophobic": "AILMFVW",   # nu le place apa: miezul proteinei, membrane
    "positive": "KRH",          # încărcați +: semnale nucleare, legare de ADN
    "negative": "DE",           # încărcați −
    "polar": "STNQ",            # polari fără sarcină
    "aromatic": "FWY",          # inel aromatic
    "small": "AGSC",            # mici, flexibili
}

N_TERM = 30      # peptidul semnal e în primii ~15–30 aminoacizi
C_TERM = 10
TM_WINDOW = 19   # un segment care traversează membrana are ~19–21 aminoacizi
TM_THRESHOLD = 1.6  # pragul clasic Kyte–Doolittle pentru o fereastră de 19
SP_WINDOW = 8    # miezul hidrofob al peptidului semnal are ~7–15 aminoacizi


def clean(seq: str) -> str:
    """Doar cele 20 de litere standard (scoate U = selenocisteină, X = necunoscut etc.)."""
    return "".join(ch for ch in seq.upper() if ch in AMINO_ACIDS)


def window_means(values: list[float], w: int) -> list[float]:
    """Media pe ferestre glisante de lungime w (ca o medie mobilă)."""
    if len(values) < w:
        return [sum(values) / len(values)] if values else [0.0]
    s = sum(values[:w])
    out = [s / w]
    for i in range(w, len(values)):
        s += values[i] - values[i - w]
        out.append(s / w)
    return out


def count_segments(means: list[float], threshold: float, w: int) -> int:
    """Numărul de regiuni distincte în care media depășește pragul.

    Ferestrele vecine peste prag (care se suprapun) formează un singur segment;
    o regiune foarte lungă e numărată ca mai multe elice lipite (~25 de aminoacizi pe elice).
    """
    count, run = 0, 0
    for m in list(means) + [float("-inf")]:
        if m > threshold:
            run += 1
        elif run:
            count += max(1, round((run + w - 1) / 25))
            run = 0
    return count


def basic_features(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    return {
        "length": len(seq),
        "log_length": math.log10(len(seq)),
        "mol_weight_kda": pa.molecular_weight() / 1000,
    }


def composition(seq: str) -> dict:
    n = len(seq)
    feats = {f"aa_{a}": seq.count(a) / n for a in AMINO_ACIDS}
    for name, letters in GROUPS.items():
        feats[f"frac_{name}"] = sum(seq.count(a) for a in letters) / n
    return feats


def physchem(seq: str) -> dict:
    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    n = len(seq)
    pct = {a: 100 * seq.count(a) / n for a in "AVIL"}
    return {
        "isoelectric_point": pa.isoelectric_point(),
        "charge_ph7": pa.charge_at_pH(7.0),
        "charge_per_residue": pa.charge_at_pH(7.0) / n,
        "gravy": sum(KYTE_DOOLITTLE[a] for a in seq) / n,
        "aromaticity": sum(seq.count(a) for a in "FWY") / n,
        "instability_index": pa.instability_index(),
        "aliphatic_index": pct["A"] + 2.9 * pct["V"] + 3.9 * (pct["I"] + pct["L"]),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet,
    }


def region_features(seq: str) -> dict:
    kd = [KYTE_DOOLITTLE[a] for a in seq]
    nterm, cterm = seq[:N_TERM], seq[-C_TERM:]
    tm_means = window_means(kd, TM_WINDOW)

    return {
        # --- capătul N (începutul): peptidul semnal
        "nterm_gravy": sum(kd[:N_TERM]) / len(nterm),
        "nterm_max_hydro_window": max(window_means(kd[:N_TERM], SP_WINDOW)),
        "nterm_frac_hydrophobic": sum(nterm.count(a) for a in GROUPS["hydrophobic"]) / len(nterm),
        "nterm_positive_first5": sum(seq[:5].count(a) for a in "KR"),
        "nterm_frac_ST": sum(nterm.count(a) for a in "ST") / len(nterm),
        # --- capătul C (sfârșitul): semnale de reținere / direcționare
        "cterm_kdel": int(bool(re.search(r"[KH]DEL$", seq))),     # rămâne în reticulul endoplasmic
        "cterm_pts1": int(bool(re.search(r"[SAC][KRH]L$", seq))),  # merge în peroxizom
        "cterm_charge": sum(cterm.count(a) for a in "KR") - sum(cterm.count(a) for a in "DE"),
        "cterm_gravy": sum(kd[-C_TERM:]) / len(cterm),
        # --- segmente transmembranare (bucăți lungi și hidrofobe)
        "tm_segments": count_segments(tm_means, TM_THRESHOLD, TM_WINDOW),
        "max_hydro_window19": max(tm_means),
        # --- semnale de localizare nucleară (NLS): grupuri de K/R
        "nls_monopartite": len(re.findall(r"(?=K[KR].[KR])", seq)),
        "nls_bipartite": len(re.findall(r"(?=[KR]{2}.{10,12}[KR]{3})", seq)),
        "max_basic_window8": max(sum(1 for a in seq[i:i + 8] if a in "KR")
                                 for i in range(max(1, len(seq) - 7))),
    }


def all_features(raw_seq: str) -> dict:
    seq = clean(raw_seq)
    if len(seq) < 30:
        raise ValueError("secvență prea scurtă")
    return {**basic_features(seq), **composition(seq), **physchem(seq), **region_features(seq)}
