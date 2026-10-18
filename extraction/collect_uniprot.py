"""Descarcă proteinele umane verificate (Swiss-Prot) din UniProt, pagină cu pagină.

Folosește API-ul REST public UniProt (https://www.uniprot.org/help/api):
- cererea întoarce maximum 500 de proteine pe pagină;
- adresa paginii următoare vine în antetul HTTP `Link` (paginare cu cursor);
- între pagini facem o pauză, iar la erori reîncercăm cu așteptare tot mai lungă.

Rezultat:
    data/raw/uniprot_human_<data>.tsv        datele brute, exact cum vin din UniProt
    data/raw/uniprot_human_<data>.meta.json  când, ce interogare, ce versiune UniProt, câte rânduri

Rulare:
    python extraction/collect_uniprot.py
    python extraction/collect_uniprot.py --limit 1000      # test rapid
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://rest.uniprot.org/uniprotkb/search"

# reviewed:true = Swiss-Prot (verificat manual), 9606 = Homo sapiens
QUERY = "reviewed:true AND organism_id:9606"

# Ce coloane cerem. NU cerem adnotările de tip „segment transmembranar” sau „peptid semnal”
# din UniProt: le calculăm noi din secvență (altfel modelul ar primi răspunsul gata făcut).
FIELDS = [
    "accession",               # ID unic, de ex. P04637
    "id",                      # nume scurt, de ex. P53_HUMAN
    "protein_name",
    "gene_primary",
    "length",
    "sequence",                # secvența de aminoacizi: din ea calculăm features
    "cc_subcellular_location", # textul din care facem eticheta
    "annotation_score",        # 1–5, cât de bine e adnotată proteina
]


def next_link(response: requests.Response) -> str | None:
    """Adresa paginii următoare, din antetul `Link: <url>; rel="next"`."""
    m = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
    return m.group(1) if m else None


def get(session: requests.Session, url: str, params: dict | None = None, retries: int = 5):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code}, reîncerc…")
        except requests.RequestException as exc:
            print(f"  eroare de rețea ({exc.__class__.__name__}), reîncerc…")
        time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16 secunde
    raise RuntimeError(f"Nu am putut descărca {url}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=QUERY)
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--pause", type=float, default=0.5, help="secunde între pagini")
    ap.add_argument("--limit", type=int, default=0, help="oprește după N rânduri (0 = toate)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    tsv_path = args.out / f"uniprot_human_{today}.tsv"

    session = requests.Session()
    session.headers["User-Agent"] = "student-ml-project (protein localization)"
    params = {"query": args.query, "fields": ",".join(FIELDS), "format": "tsv", "size": args.page_size}

    r = get(session, URL, params)
    total = int(r.headers.get("X-Total-Results", 0))
    release = r.headers.get("X-UniProt-Release", "?")
    print(f"UniProt release {release}: {total} proteine pentru „{args.query}”")

    n_rows, page = 0, 1
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        while True:
            lines = r.text.rstrip("\n").split("\n")
            header, rows = lines[0], lines[1:]
            if page == 1:
                f.write(header + "\n")
            for line in rows:
                f.write(line + "\n")
            n_rows += len(rows)
            print(f"  pagina {page}: {n_rows}/{total}")
            url = next_link(r)
            if not url or (args.limit and n_rows >= args.limit):
                break
            time.sleep(args.pause)
            r = get(session, url)
            page += 1

    meta = {
        "downloaded": dt.datetime.now().isoformat(timespec="seconds"),
        "source": URL,
        "query": args.query,
        "fields": FIELDS,
        "uniprot_release": release,
        "total_results": total,
        "rows_written": n_rows,
        "file": tsv_path.name,
    }
    tsv_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Salvat: {tsv_path} ({n_rows} rânduri)")


if __name__ == "__main__":
    sys.exit(main())
