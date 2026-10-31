from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import TOP_K
from preprocessing import tokenize_and_process, preprocess_text, build_document_corpus_series
"""Давлатов Мансур МТ-202"""

class WordOverlapSearcher:
    """Базовый поиск по пересечению множества слов запроса и документа."""

    def __init__(self, lemmatize: bool = True, remove_stopwords: bool = True):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.doc_ids: List[str] = []
        self.doc_token_sets: List[set] = []

    def fit(self, documents_df: pd.DataFrame) -> "WordOverlapSearcher":
        self.doc_ids = documents_df["document_id"].tolist()
        raw_texts = build_document_corpus_series(documents_df)
        self.doc_token_sets = [
            set(tokenize_and_process(text, self.lemmatize, self.remove_stopwords))
            for text in raw_texts
        ]
        return self

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Возвращает TOP-K пар (document_id, score). Корректно обрабатывает пустой запрос."""
        q_tokens = set(tokenize_and_process(query, self.lemmatize, self.remove_stopwords))
        if not q_tokens:
            return [(doc_id, 0.0) for doc_id in self.doc_ids[:top_k]]

        scores = np.array(
            [float(len(q_tokens & doc_set)) for doc_set in self.doc_token_sets],
            dtype=float,
        )
        top_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]


class TfidfSearcher:
    """Поисковая система на основе TF-IDF и косинусного сходства."""

    def __init__(
        self,
        lemmatize: bool = True,
        remove_stopwords: bool = True,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.doc_ids: List[str] = []
        self.doc_matrix = None

    def fit(self, documents_df: pd.DataFrame) -> "TfidfSearcher":
        """Обучает векторизатор на корпусе документов (fit_transform)."""
        self.doc_ids = documents_df["document_id"].tolist()
        raw_texts = build_document_corpus_series(documents_df)
        processed_docs = [
            preprocess_text(t, self.lemmatize, self.remove_stopwords)
            for t in raw_texts
        ]
        self.doc_matrix = self.vectorizer.fit_transform(processed_docs)
        return self

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Переводит запрос в пространство корпуса (transform) и возвращает TOP-K документов."""
        processed_q = preprocess_text(query, self.lemmatize, self.remove_stopwords)
        if not processed_q.strip():
            return [(doc_id, 0.0) for doc_id in self.doc_ids[:top_k]]

        q_vec = self.vectorizer.transform([processed_q])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(-sims, kind="stable")[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in top_indices]

    def search_batch(self, queries_df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
        """Выполняет пакетный поиск по всем запросам и формирует таблицу формата search_results.csv."""
        rows = []
        for _, row in queries_df.iterrows():
            qid = row["query_id"]
            q_text = row["query"]
            top_results = self.search(q_text, top_k=top_k)
            for rank_idx, (doc_id, score) in enumerate(top_results, start=1):
                rows.append(
                    {
                        "query_id": qid,
                        "rank": rank_idx,
                        "document_id": doc_id,
                        "score": round(score, 6),
                    }
                )
        return pd.DataFrame(rows)
