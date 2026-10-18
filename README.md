# Localizarea subcelulară a proteinelor din secvență

Proiect pentru *Învățare automată și procesarea datelor* (Master Știința Datelor, 2026–2027).

**Întrebarea:** în ce parte a celulei lucrează o proteină umană (nucleu, citoplasmă,
membrană sau în afara celulei), dacă știm doar secvența ei de aminoacizi?

**Datele:** set construit de noi. Proteinele umane verificate manual (Swiss-Prot) sunt
descărcate prin API-ul public UniProt. Eticheta vine din localizările demonstrate
experimental, iar cele 53 de features sunt calculate de noi din secvență. Setul de date
nu există gata făcut în această formă.

## Instalare

```bash
pip install -r requirements.txt
```

## Pași

```bash
python extraction/collect_uniprot.py     # 1. descarcă datele brute (~20.000 proteine, câteva minute)
python extraction/build_dataset.py       # 2. etichete + features + grupuri → data/processed/dataset.csv
python analysis/eda.py                   # 3. grafice exploratorii → results/eda/
python scripts/train.py                  # 4. modele + evaluare → results/
pytest                                   # teste (nu au nevoie de internet)
```

## Datele

| | |
|---|---|
| Sursă | [UniProt](https://www.uniprot.org/), prin [API-ul REST](https://www.uniprot.org/help/api) |
| Interogare | `reviewed:true AND organism_id:9606` (Swiss-Prot, *Homo sapiens*) |
| Clase | Nucleus, Cytoplasm, Membrane, Extracellular |
| Etichetă | doar locații cu dovadă experimentală (ECO:0000269 / ECO:0000314); proteinele cu mai multe clase sunt scoase |
| Features | 53, calculate din secvență (vezi tabelul de mai jos) |
| Împărțire | 80/20 pe grupuri de proteine asemănătoare (3-meri, cosinus ≥ 0,4) |

Fiecare descărcare salvează și un fișier `.meta.json` (data, interogarea, versiunea UniProt,
numărul de rânduri), ca rezultatul să poată fi refăcut. `data/processed/label_report.csv`
arată câte proteine au fost scoase și de ce.

### Features

| Grup | Features | Ce indică |
|---|---|---|
| De bază | lungime, log(lungime), masă (kDa) | proteinele mari rar sunt secretate |
| Compoziție | fracția celor 20 de aminoacizi, 6 grupe (hidrofobi, +, −, polari, aromatici, mici) | profilul general al proteinei |
| Fizico-chimice | pI, sarcină la pH 7, sarcină/aminoacid, GRAVY, aromaticitate, instabilitate, indice alifatic, structură secundară | mediul în care trăiește proteina |
| Capătul N | hidrofobicitatea primilor 30 de aminoacizi, fereastra hidrofobă maximă, K/R în primii 5, fracția S/T | peptid semnal → secretată sau membrană |
| Capătul C | KDEL/HDEL, PTS1 (SKL), sarcina și hidrofobicitatea ultimilor 10 | semnale de reținere și ancore |
| Transmembranar | numărul de segmente hidrofobe de 19 aminoacizi, fereastra maximă | proteină de membrană |
| Nuclear (NLS) | motive K(K/R)x(K/R), NLS bipartit, max. K/R în 8 aminoacizi | intrare în nucleu |

## Modele și evaluare

- **Baseline:** clasa majoritară.
- **Modele:** regresie logistică, Random Forest, Gradient Boosting (toate cu ponderi pe clase).
- **Metrică principală:** macro-F1 (media F1 pe clase), pentru că clasele nu sunt egale.
- **Validare:** 5 părți pe train (pe grupuri); setul de test e folosit o singură dată.
- **Interpretare:** importanța features-urilor prin permutare.

## Rezultate

*de completat după rularea pe datele reale*

## Structura

```
extraction/
  collect_uniprot.py   descărcare din UniProt (paginare cu cursor, reîncercări, log)
  build_dataset.py     etichete, filtre, features, grupuri → setul final
localization/
  labels.py            textul de localizare → clasă
  features.py          calculul celor 53 de features
  split.py             grupuri de proteine asemănătoare
analysis/eda.py        analiza exploratorie
scripts/train.py       antrenare și evaluare
tests/                 teste
data/raw/              date brute (fișierul .tsv nu se urcă pe git, doar .meta.json)
data/processed/        setul final de date
results/               metrici și grafice
```

## Referințe

- The UniProt Consortium (2025). UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research*, 53(D1).
- Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. *J. Mol. Biol.*, 157(1).
- Guruprasad, K., Reddy, B. V., & Pandit, M. W. (1990). Correlation between stability of a protein and its dipeptide composition. *Protein Engineering*, 4(2).
- Ikai, A. (1980). Thermostability and aliphatic index of globular proteins. *J. Biochem.*, 88(6).
- Almagro Armenteros, J. J. et al. (2017). DeepLoc: prediction of protein subcellular localization using deep learning. *Bioinformatics*, 33(21).
