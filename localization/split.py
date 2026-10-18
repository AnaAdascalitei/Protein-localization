"""Grupuri de proteine asemănătoare, ca train și test să nu conțină „rude”.

Problema: proteinele din aceeași familie (de ex. zeci de receptori olfactivi) au secvențe
aproape identice și aceeași localizare. Dacă una e în train și sora ei în test, modelul
„recunoaște” proteina în loc să învețe regula, iar scorul pe test e umflat.

Soluția (simplă, fără programe externe precum CD-HIT):
1. fiecare secvență devine un vector cu numărul de apariții al fiecărui „cuvânt” de 3 litere
   (3-mer: MKT, KTA, TAY ...); există 20^3 = 8.000 de cuvinte posibile;
2. similaritatea dintre două proteine = cosinusul dintre vectori (0 = nimic comun, 1 = identic);
3. legăm proteinele cu similaritate peste prag și luăm componentele conexe ca grupuri;
4. la împărțire, un grup întreg merge fie în train, fie în test.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def similarity_groups(sequences: list[str], threshold: float = 0.4, k: int = 3) -> np.ndarray:
    vec = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
    X = normalize(vec.fit_transform(sequences)).tocsr()   # fiecare rând are lungimea 1
    rows, cols = [], []
    for start in range(0, X.shape[0], 1000):              # pe bucăți, ca să nu umplem memoria
        block = (X[start:start + 1000] @ X.T).toarray()    # cosinus = produs scalar
        r, c = np.nonzero(block >= threshold)
        rows.append(r + start)
        cols.append(c)
    n = X.shape[0]
    graph = coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n))
    _, groups = connected_components(graph, directed=False)
    return groups
