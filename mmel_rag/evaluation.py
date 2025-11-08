from __future__ import annotations

import csv
import math
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
from langchain_core.documents import Document

from .config import DOCS_DIR, EVAL_DIR, DEVSET_CSV
from .retrieval import get_docs


# Normalise un chemin de source pour pouvoir comparer les chemins proprement
def normalize_source(s) :
    s = str(s or "").strip()
    s = s.replace("\\", "/")
    parts = s.split("/")
    if "Docs" in parts:
        idx = parts.index("Docs")
        parts = parts[idx:]
    return "/".join(parts)


# Nettoie légèrement le texte (questions / réponses attendues)
def clean(s):
    return re.sub(r"\s+", " ", str(s or "").strip())


# Charge le CSV du devset en étant tolérant sur les colonnes et les espaces
def load_devset_csv_tolerant(path: str | None = None) :
    if path is None:
        path = DEVSET_CSV

    dev: List[Dict[str, Any]] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # gold: "Docs\\CS-25 (Amendment 27).pdf:430;Docs\\...:412"
            gold_pairs: List[Tuple[str, str]] = []
            raw = clean(row.get("gold", ""))
            if raw:
                for part in raw.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    if ":" in part:
                        src, pg = part.split(":", 1)
                        gold_pairs.append(
                            (normalize_source(src), clean(pg))
                        )

            must = [
                w.strip()
                for w in (row.get("must_include") or "").split("|")
                if w.strip()
            ]

            dev.append(
                {
                    "q": clean(row.get("q", "")),
                    "gold": gold_pairs,
                    "must_include": must,
                    "lang": (row.get("lang") or "en").strip(),
                }
            )

    return dev


DEVSET = load_devset_csv_tolerant()
print("DEVSET questions:", len(DEVSET))


# Vérifie si un document fait partie des réponses gold (tolérance sur la page)
def doc_matches_gold(doc: Document, gold_pairs, tol: int = 3) :
    if not gold_pairs:
        return False

    src = normalize_source(doc.metadata.get("source", "?"))
    page_raw = doc.metadata.get("page", "?")

    try:
        pdoc = int(str(page_raw))
    except Exception:
        pdoc = None

    for (gsrc, gpage) in gold_pairs:
        # gsrc vient déjà normalisé mais on repasse par normalize_source par sécurité
        if src != normalize_source(gsrc):
            continue

        try:
            gp = int(str(gpage))
            if pdoc is not None and abs(pdoc - gp) <= tol:
                return True
        except Exception:
            # fallback sur comparaison brute si gpage n'est pas convertible en int
            if str(page_raw) == str(gpage):
                return True

    return False


# Évalue un mode de retrieval sur le devset (hit@k, MRR, nDCG@5)
def eval_retriever(
    devset: List[Dict[str, Any]],
    mode: str,
    k_list: Tuple[int, ...] = (1, 3, 5),
    show_errors: bool = False,
    k_eval: int = 5,
    page_tol: int = 3,
) :
    # Normalise un texte (minuscule + espaces) pour comparer proprement
    def normalize_text(s) :
        s = (s or "").lower()
        return re.sub(r"\s+", " ", s).strip()

    # Heuristique de secours : le texte contient toutes les must_include
    def doc_contains_all_keywords(doc, kws):
        if not kws:
            return False
        t = normalize_text(doc.page_content)
        return all(normalize_text(k) in t for k in kws)

    # Gain binaire pour la nDCG (1 si bonne page, 0 sinon)
    def rel(doc, item) :
        if doc_matches_gold(doc, item.get("gold"), tol=page_tol):
            return 1
        if doc_contains_all_keywords(doc, item.get("must_include")):
            return 1
        return 0

    # Discounted Cumulative Gain
    def dcg(gains, k: int = 5):
        return sum(gains[i] / math.log2(i + 2) for i in range(min(k, len(gains))))

    hits = {k: 0 for k in k_list}
    rr = 0.0
    nd = 0.0
    per: List[Dict[str, Any]] = []
    n = len(devset)

    for it in devset:
        q = it["q"]
        docs = get_docs(q, mode=mode, k=k_eval)

        gains: List[int] = []
        first = None
        for r, d in enumerate(docs, 1):
            g = rel(d, it)
            gains.append(g)
            if g and first is None:
                first = r

        for k in k_list:
            if any(gains[:k]):
                hits[k] += 1
        if first:
            rr += 1.0 / first

        ideal = sorted(gains, reverse=True)
        idcg = dcg(ideal, k=5)
        nd += dcg(gains, k=5) / (idcg if idcg > 0 else 1)

        per.append(
            {
                "q": q,
                "first_hit_rank": first or 0,
                "gains": gains,
                "top_preview": [
                    {
                        "rank": i + 1,
                        "source": d.metadata.get("source", "?"),
                        "page": d.metadata.get("page", "?"),
                        "excerpt": d.page_content[:180].replace("\n", " "),
                    }
                    for i, d in enumerate(docs)
                ],
            }
        )

    summary = {
        "mode": mode,
        "n": n,
        "hit@1": round(hits.get(1, 0) / n, 3),
        "hit@3": round(hits.get(3, 0) / n, 3),
        "hit@5": round(hits.get(5, 0) / n, 3),
        "mrr": round(rr / n, 3),
        "ndcg@5": round(nd / n, 3),
        "page_tol": page_tol,
    }

    if show_errors:
        fails = [r for r in per if r["first_hit_rank"] in (None, 0)]
        if fails:
            print(f"\nErreurs ({len(fails)}/{n}) — exemples :")
            for r in fails[:5]:
                print("Q:", r["q"])
                for tp in r["top_preview"][:3]:
                    print(
                        f"  - [{tp['rank']}] {tp['source']} p.{tp['page']} "
                        f"{tp['excerpt'][:160]}…"
                    )
                print()
    return summary, per


# Convertit les résultats par question en DataFrame détaillée (une ligne par doc)
def perq_to_df(perq: List[Dict[str, Any]], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in perq:
        for tp in r["top_preview"]:
            rows.append(
                {
                    "mode": mode,
                    "q": r["q"],
                    "first_hit_rank": r["first_hit_rank"],
                    "rank": tp["rank"],
                    "source": tp["source"],
                    "page": tp["page"],
                    "excerpt": tp["excerpt"],
                }
            )
    return pd.DataFrame(rows)


# Lance l'évaluation complète (vector, bm25translate, hybrid) et exporte les CSV
def run_full_eval(
    devset: List[Dict[str, Any]] | None = None,
    export: bool = True,
):
    # mêmes paramètres que dans le notebook : k_eval=8, page_tol=3, K_LIST=(1,3,5)
    if devset is None:
        devset = DEVSET

    K_LIST = (1, 3, 5)

    sum_vec, per_vec = eval_retriever(
        devset, "vector", k_list=K_LIST, show_errors=True, k_eval=8, page_tol=3
    )
    sum_bm, per_bm = eval_retriever(
        devset, "bm25translate", k_list=K_LIST, show_errors=True, k_eval=8, page_tol=3
    )
    sum_hy, per_hy = eval_retriever(
        devset, "hybrid", k_list=K_LIST, show_errors=True, k_eval=8, page_tol=3
    )

    print("\n— RÉSUMÉ —")
    print(
        f"vector         | hit@1={sum_vec['hit@1']:.3f} | hit@5={sum_vec['hit@5']:.3f} "
        f"| MRR={sum_vec['mrr']:.3f} | nDCG@5={sum_vec['ndcg@5']:.3f}"
    )
    print(
        f"bm25translate  | hit@1={sum_bm['hit@1']:.3f}  | hit@5={sum_bm['hit@5']:.3f}  "
        f"| MRR={sum_bm['mrr']:.3f}  | nDCG@5={sum_bm['ndcg@5']:.3f}"
    )
    print(
        f"hybrid_rrf     | hit@1={sum_hy['hit@1']:.3f}  | hit@5={sum_hy['hit@5']:.3f}  "
        f"| MRR={sum_hy['mrr']:.3f}  | nDCG@5={sum_hy['ndcg@5']:.3f}"
    )

    if export:
        pd.DataFrame(
            [
                {"mode": "vector", **sum_vec},
                {"mode": "bm25translate", **sum_bm},
                {"mode": "hybrid_rrf", **sum_hy},
            ]
        ).to_csv(EVAL_DIR / "summary_retrieval.csv", index=False)

        df_per = pd.concat(
            [
                perq_to_df(per_vec, "vector"),
                perq_to_df(per_bm, "bm25translate"),
                perq_to_df(per_hy, "hybrid_rrf"),
            ],
            ignore_index=True,
        )
        df_per.to_csv(EVAL_DIR / "per_question_results.csv", index=False)

    return {
        "vector": (sum_vec, per_vec),
        "bm25translate": (sum_bm, per_bm),
        "hybrid": (sum_hy, per_hy),
    }
