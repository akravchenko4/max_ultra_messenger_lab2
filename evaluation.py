#Антонов Александр мт-202
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from config import TOP_K


def compute_metrics_at_k(
    searcher,
    queries_df: pd.DataFrame,
    dev_relevance_df: pd.DataFrame,
    k: int = TOP_K,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Вычисляет Hit@K и MRR@K только на открытых запросах из dev_relevance.csv.
    Возвращает (hit_at_k, mrr_at_k, детальная_таблица_по_запросам).
    """
    merged = dev_relevance_df.merge(queries_df, on="query_id", how="inner")
    hits = []
    reciprocal_ranks = []
    details = []

    for _, row in merged.iterrows():
        qid = row["query_id"]
        rel_doc_id = row["relevant_document_id"]
        q_text = row["query"]

        top_k_results = searcher.search(q_text, top_k=k)
        top_doc_ids = [doc_id for doc_id, _ in top_k_results]

        if rel_doc_id in top_doc_ids:
            rank = top_doc_ids.index(rel_doc_id) + 1
            hit = 1.0
            rr = 1.0 / rank
        else:
            rank = None
            hit = 0.0
            rr = 0.0

        hits.append(hit)
        reciprocal_ranks.append(rr)
        details.append(
            {
                "query_id": qid,
                "relevant_document_id": rel_doc_id,
                "found_rank_in_top5": rank if rank is not None else "MISS (>5)",
                "hit_at_5": hit,
                "rr_at_5": round(rr, 4),
                "top1_predicted": top_doc_ids[0] if top_doc_ids else None,
                "top5_predicted": ", ".join(top_doc_ids),
            }
        )

    hit_at_k = float(np.mean(hits)) if hits else 0.0
    mrr_at_k = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return round(hit_at_k, 4), round(mrr_at_k, 4), pd.DataFrame(details)
