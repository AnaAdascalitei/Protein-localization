"""Transformă textul de localizare din UniProt într-o etichetă (clasă).

Exemplu de text din UniProt:
    SUBCELLULAR LOCATION: Nucleus {ECO:0000269|PubMed:15710747}. Cytoplasm
    {ECO:0000250|UniProtKB:P02340}. Note=Translocates to the nucleus...;
    SUBCELLULAR LOCATION: [Isoform 2]: Cytoplasm.

Pași:
1. Păstrăm doar primul bloc „SUBCELLULAR LOCATION” (forma principală a proteinei,
   nu izoformele) și tăiem notele libere („Note=...”).
2. Despărțim în locații. Fiecare are o dovadă (cod ECO):
   ECO:0000269 / ECO:0000314 = demonstrat experimental; restul = dedus (similaritate etc.).
3. Fiecare locație intră într-una din clase: Nucleus, Cytoplasm, Membrane, Extracellular
   sau Other (mitocondrie, reticul endoplasmic, Golgi, lizozom ...).
4. Păstrăm proteina doar dacă TOATE locațiile (cu dovada cerută) dau o singură clasă
   din cele 4. Proteinele cu mai multe clase (de ex. nucleu + citoplasmă) sunt scoase,
   ca eticheta să fie clară.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CLASSES = ["Nucleus", "Cytoplasm", "Membrane", "Extracellular"]
EXPERIMENTAL = ("ECO:0000269", "ECO:0000314")


@dataclass
class Location:
    term: str           # de ex. "Cell membrane"
    topology: str       # de ex. "Multi-pass membrane protein" (poate fi gol)
    evidence: str       # textul din acolade, de ex. "ECO:0000269|PubMed:123"

    @property
    def experimental(self) -> bool:
        return any(code in self.evidence for code in EXPERIMENTAL)


def main_block(text: str) -> str:
    """Primul bloc fără „[Isoform …]”, fără prefix și fără note."""
    if not isinstance(text, str) or not text.strip():
        return ""
    blocks = [b.strip() for b in text.split("SUBCELLULAR LOCATION:") if b.strip()]
    if not blocks:
        return ""
    block = next((b for b in blocks if not b.startswith("[Isoform")), blocks[0])
    block = re.sub(r"^\[Isoform[^\]]*\]:\s*", "", block)
    block = block.split("Note=")[0]
    return block.strip(" ;")


def _split_outside_braces(text: str, sep: str) -> list[str]:
    """Împarte după `sep`, dar nu în interiorul acoladelor {…} cu dovezi."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "{") - (ch == "}")
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_locations(text: str) -> list[Location]:
    out = []
    # „Cell membrane {ECO..}; Multi-pass membrane protein {ECO..}. Cytoplasm {ECO..}.”
    for segment in _split_outside_braces(main_block(text), "."):
        parts = _split_outside_braces(segment, ";")
        first = parts[0]
        evidence = " ".join(re.findall(r"\{([^}]*)\}", first))
        term = re.sub(r"\{[^}]*\}", "", first).strip()
        topology = "; ".join(re.sub(r"\{[^}]*\}", "", p).strip() for p in parts[1:])
        if term:
            out.append(Location(term=term, topology=topology, evidence=evidence))
    return out


def location_class(loc: Location) -> str | None:
    """Clasa unei locații. None = ignorată (nu ajută și nu încurcă eticheta)."""
    term, topo = loc.term.lower(), loc.topology.lower()
    if "membrane" in term:
        # proteinele periferice doar „stau lipite” de membrană, nu o traversează:
        # din secvență arată ca proteinele citoplasmatice, deci nu le folosim
        if "peripheral" in topo:
            return None
        return "Membrane"
    if term.startswith(("nucleus", "chromosome")):
        return "Nucleus"
    if term.startswith(("cytoplasm", "cytosol")):
        return "Cytoplasm"
    if term.startswith("secreted"):
        return "Extracellular"
    return "Other"


def label(text: str, evidence: str = "experimental") -> tuple[str | None, str]:
    """(clasa, motiv). Clasa e None dacă proteina nu intră în setul de date."""
    locs = parse_locations(text)
    if evidence == "experimental":
        locs = [l for l in locs if l.experimental]
    if not locs:
        return None, "fără locație" if evidence == "any" else "fără dovadă experimentală"
    classes = {c for c in (location_class(l) for l in locs) if c}
    if not classes:
        return None, "doar membrană periferică"
    if "Other" in classes:
        return None, "alt compartiment"
    if len(classes) > 1:
        return None, "mai multe clase"
    return classes.pop(), "ok"
