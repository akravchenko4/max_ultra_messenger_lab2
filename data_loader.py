"""Соболев Вадим МТ-202"""
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from config import DOCUMENTS_PATH, QUERIES_PATH, DEV_RELEVANCE_PATH, CSV_ENCODING


def load_datasets(
    docs_path: Path = DOCUMENTS_PATH,
    queries_path: Path = QUERIES_PATH,
    dev_path: Path = DEV_RELEVANCE_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает CSV-файлы корпуса, запросов и эталонной разметки."""
    documents = pd.read_csv(docs_path, encoding=CSV_ENCODING)
    queries = pd.read_csv(queries_path, encoding=CSV_ENCODING)
    dev_relevance = pd.read_csv(dev_path, encoding=CSV_ENCODING)
    return documents, queries, dev_relevance


def inspect_corpus(
    documents: pd.DataFrame,
    queries: pd.DataFrame,
    dev_relevance: pd.DataFrame,
) -> Dict[str, Any]:
    """Исследует типы данных, пропуски, дубликаты и уникальность идентификаторов."""
    report = {
        "num_documents": len(documents),
        "doc_columns": list(documents.columns),
        "doc_dtypes": documents.dtypes.astype(str).to_dict(),
        "doc_missing": documents.isnull().sum().to_dict(),
        "unique_doc_ids": int(documents["document_id"].nunique()),
        "duplicate_doc_rows": int(documents.duplicated().sum()),
        "duplicate_doc_texts": int(documents["text"].duplicated().sum()),
        "sections_distribution": documents["section"].value_counts().to_dict(),
        "num_queries": len(queries),
        "query_columns": list(queries.columns),
        "query_missing": queries.isnull().sum().to_dict(),
        "unique_query_ids": int(queries["query_id"].nunique()),
        "difficulty_distribution": queries["difficulty"].value_counts().to_dict(),
        "num_dev_pairs": len(dev_relevance),
    }
    return report


def print_corpus_report(report: Dict[str, Any]) -> None:
    """Выводит сводку по результатам исследования корпуса."""
    print("=== 1. ИССЛЕДОВАНИЕ КОРПУСА И ЗАПРОСОВ ===")
    print(f"Количество документов: {report['num_documents']} (уникальных ID: {report['unique_doc_ids']})")
    print(f"Поля документов: {report['doc_columns']}")
    print(f"Типы полей: {report['doc_dtypes']}")
    print(f"Пропуски в документах: {report['doc_missing']}")
    print(f"Полных дубликатов строк: {report['duplicate_doc_rows']}, дубликатов текста: {report['duplicate_doc_texts']}")
    print(f"Распределение по разделам: {report['sections_distribution']}")
    print(f"Количество запросов: {report['num_queries']} (уникальных ID: {report['unique_query_ids']})")
    print(f"Типы сложности запросов: {report['difficulty_distribution']}")
    print(f"Количество размеченных запросов в dev_relevance.csv: {report['num_dev_pairs']}\n")
