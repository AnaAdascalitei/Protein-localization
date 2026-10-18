"""Teste fără internet: rulați cu `pytest`."""
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from localization.features import all_features, clean, count_segments, window_means  # noqa: E402
from localization.labels import label, parse_locations  # noqa: E402
from localization.split import similarity_groups  # noqa: E402

EXP = "{ECO:0000269|PubMed:123, ECO:0000269|PubMed:456}"
INF = "{ECO:0000250|UniProtKB:P12345}"


# ------------------------------------------------------------------ etichete
def test_parse_membrane_with_topology_and_isoforms():
    text = (f"SUBCELLULAR LOCATION: Cell membrane {EXP}; Multi-pass membrane protein {{ECO:0000255}}. "
            f"Note=Moves to the cell surface. {EXP}.; SUBCELLULAR LOCATION: [Isoform 2]: Cytoplasm {EXP}.")
    locs = parse_locations(text)
    assert [(l.term, l.topology, l.experimental) for l in locs] == [
        ("Cell membrane", "Multi-pass membrane protein", True)]
    assert label(text) == ("Membrane", "ok")


def test_label_rules():
    assert label(f"SUBCELLULAR LOCATION: Nucleus, nucleolus {EXP}.")[0] == "Nucleus"
    assert label(f"SUBCELLULAR LOCATION: Secreted {EXP}.")[0] == "Extracellular"
    # nucleu + citoplasmă = ambiguu → scos
    assert label(f"SUBCELLULAR LOCATION: Nucleus {EXP}. Cytoplasm {EXP}.") == (None, "mai multe clase")
    # dovada doar dedusă → scoasă implicit, păstrată cu evidence="any"
    assert label(f"SUBCELLULAR LOCATION: Cytoplasm {INF}.")[0] is None
    assert label(f"SUBCELLULAR LOCATION: Cytoplasm {INF}.", "any")[0] == "Cytoplasm"
    # locația dedusă nu strică eticheta experimentală
    assert label(f"SUBCELLULAR LOCATION: Nucleus {EXP}. Cytoplasm {INF}.")[0] == "Nucleus"
    assert label(f"SUBCELLULAR LOCATION: Mitochondrion matrix {EXP}.") == (None, "alt compartiment")
    assert label(f"SUBCELLULAR LOCATION: Cell membrane {EXP}; Peripheral membrane protein {EXP}.")[0] is None
    assert label(float("nan"))[0] is None


# ------------------------------------------------------------------ features
def test_features_match_definitions():
    seq = "MKKRKVLLLLLLAAAVVVSSTTDEKDEL"
    f = all_features(seq * 2)
    assert f["length"] == len(seq) * 2
    assert abs(sum(f[f"aa_{a}"] for a in "ACDEFGHIKLMNPQRSTVWY") - 1) < 1e-9
    assert f["cterm_kdel"] == 1 and f["cterm_pts1"] == 0
    assert all_features("A" * 40 + "SKL")["cterm_pts1"] == 1
    assert f["nls_monopartite"] >= 1


def test_gravy_matches_biopython():
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    assert abs(all_features(seq)["gravy"] - ProteinAnalysis(seq).gravy()) < 1e-9


def test_transmembrane_segments_are_counted():
    rng = random.Random(0)
    polar = lambda n: "".join(rng.choice("DEKRNQSTG") for _ in range(n))  # noqa: E731
    tm = "LLIVALFLAVILLGFIVAL"  # 19 aminoacizi hidrofobi
    seq = polar(40) + tm + polar(30) + tm + polar(30) + tm + polar(40)
    assert all_features(seq)["tm_segments"] == 3
    assert all_features(polar(200))["tm_segments"] == 0


def test_window_helpers():
    assert window_means([1, 1, 3, 3], 2) == [1, 2, 3]
    assert count_segments([0, 2, 2, 2, 0, 0, 2], 1, 2) == 2
    assert clean("MKU*xA") == "MKA"


def test_similarity_groups_join_near_copies():
    rng = random.Random(1)
    a = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(300))
    b = a[:150] + "W" + a[151:]  # o singură mutație
    c = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(300))
    g = similarity_groups([a, b, c])
    assert g[0] == g[1] != g[2]


# ------------------------------------------------------------------ cap-coadă pe date sintetice
def fake_uniprot(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    """Proteine inventate, cu semnalele tipice fiecărei clase, în formatul TSV UniProt."""
    rng = random.Random(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rand = lambda n: "".join(rng.choice(aa) for _ in range(n))  # noqa: E731
    hydro = lambda n: "".join(rng.choice("LIVFAM") for _ in range(n))  # noqa: E731
    make = {
        "Nucleus": lambda: "M" + rand(80) + "PKKKRKV" + rand(150) + "KRKR" + rand(60),
        "Cytoplasm": lambda: "M" + rand(rng.randint(200, 400)),
        "Membrane": lambda: "M" + rand(40) + "".join(hydro(20) + rand(25) for _ in range(rng.randint(1, 7))),
        "Extracellular": lambda: "MKR" + hydro(14) + "AGA" + rand(200).replace("C", "") + "CC" * 8,
    }
    term = {"Nucleus": "Nucleus", "Cytoplasm": "Cytoplasm, cytosol",
            "Membrane": "Cell membrane {ECO:0000269|PubMed:1}; Multi-pass membrane protein",
            "Extracellular": "Secreted"}
    rows = []
    for cls, gen in make.items():
        for i in range(n_per_class):
            s = gen()
            loc = term[cls] if "{" in term[cls] else f"{term[cls]} {EXP}"
            rows.append({"Entry": f"{cls[:2].upper()}{i:04d}", "Entry Name": f"X{i}_HUMAN", "Protein names": "x",
                         "Gene Names (primary)": "G", "Length": str(len(s)), "Sequence": s,
                         "Subcellular location [CC]": f"SUBCELLULAR LOCATION: {loc}.", "Annotation": "5"})
    rows.append({**rows[0], "Entry": "DUP1"})  # duplicat exact
    return pd.DataFrame(rows)


def test_end_to_end_on_synthetic_data(tmp_path):
    pytest.importorskip("sklearn")
    import importlib
    build = importlib.import_module("extraction.build_dataset").build
    data, report = build(fake_uniprot())
    assert len(data) == 240 and "duplicat" in set(report["motiv"])
    assert data.shape[1] - 6 >= 50
    path = tmp_path / "dataset.csv"
    data.to_csv(path, index=False)

    train = importlib.import_module("scripts.train")
    res = train.main(path, tmp_path / "results")
    assert res[res["_best_model"]]["test_macro_f1"] > 0.8
    assert (tmp_path / "results" / "importance.png").exists()
