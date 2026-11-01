import pandas as pd
from config import (
    CONFIGURATIONS,
    BEST_CONFIGURATION,
    SEARCH_RESULTS_PATH,
    EXPERIMENT_METRICS_PATH,
    CSV_ENCODING,
    TOP_K,
)
from data_loader import load_datasets, inspect_corpus, print_corpus_report
from search_models import WordOverlapSearcher, TfidfSearcher
from evaluation import compute_metrics_at_k
"""Исаев Максим МТ-202"""

def main() -> None:
    # 1. Загрузка и исследование корпуса
    documents_df, queries_df, dev_relevance_df = load_datasets()
    corpus_report = inspect_corpus(documents_df, queries_df, dev_relevance_df)
    print_corpus_report(corpus_report)

    # 2. Проверка базового поиска по совпадению слов (включая тест пустого запроса)
    print("=== 2. БАЗОВЫЙ ПОИСК ПО СОВПАДЕНИЮ СЛОВ ===")
    baseline_raw = WordOverlapSearcher(lemmatize=False, remove_stopwords=False).fit(documents_df)
    empty_check = baseline_raw.search("", top_k=TOP_K)
    print(f"Проверка пустого запроса (TOP-5 со score=0.0): {empty_check[:2]} ...")

    b_hit_raw, b_mrr_raw, _ = compute_metrics_at_k(baseline_raw, queries_df, dev_relevance_df, k=TOP_K)
    print(f"Базовый поиск (без лемматизации): Hit@5 = {b_hit_raw:.4f}, MRR@5 = {b_mrr_raw:.4f}")

    baseline_lem = WordOverlapSearcher(lemmatize=True, remove_stopwords=True).fit(documents_df)
    b_hit_lem, b_mrr_lem, _ = compute_metrics_at_k(baseline_lem, queries_df, dev_relevance_df, k=TOP_K)
    print(f"Базовый поиск (с лемматизацией и без стоп-слов): Hit@5 = {b_hit_lem:.4f}, MRR@5 = {b_mrr_lem:.4f}\n")

    # 3. Сравнение 4 конфигураций TF-IDF + cosine similarity (A, B, C, D)
    print("=== 3. СРАВНЕНИЕ ЧЕТЫРЁХ КОНФИГУРАЦИЙ TF-IDF (A, B, C, D) ===")
    metrics_rows = []
    trained_models = {}
    detailed_reports = {}

    for cfg_key, cfg_params in CONFIGURATIONS.items():
        searcher = TfidfSearcher(
            lemmatize=cfg_params["lemmatize"],
            remove_stopwords=cfg_params["remove_stopwords"],
            ngram_range=cfg_params["ngram_range"],
        ).fit(documents_df)

        hit_5, mrr_5, details_df = compute_metrics_at_k(
            searcher, queries_df, dev_relevance_df, k=TOP_K
        )
        trained_models[cfg_key] = searcher
        detailed_reports[cfg_key] = details_df

        metrics_rows.append(
            {
                "configuration": cfg_key,
                "hit_at_5": f"{hit_5:.4f}",
                "mrr_at_5": f"{mrr_5:.4f}",
            }
        )
        print(f"Конфигурация {cfg_key} ({cfg_params['description']}): Hit@5 = {hit_5:.4f}, MRR@5 = {mrr_5:.4f}")

    # 4. Сохранение experiment_metrics.csv
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(EXPERIMENT_METRICS_PATH, index=False, encoding=CSV_ENCODING)
    print(f"\nФайл метрик сохранён: {EXPERIMENT_METRICS_PATH}")

    # 5. Детальный разбор рангов на 8 открытых запросах для лучшей конфигурации (C)
    print(f"\n=== 4. ДЕТАЛИЗАЦИЯ ПО 8 ОТКРЫТЫМ ЗАПРОСАМ (КОНФИГУРАЦИЯ {BEST_CONFIGURATION}) ===")
    print(detailed_reports[BEST_CONFIGURATION].to_string(index=False))

    # 6. Сохранение search_results.csv (30 запросов * 5 документов = 150 строк)
    best_model = trained_models[BEST_CONFIGURATION]
    search_results_df = best_model.search_batch(queries_df, top_k=TOP_K)
    search_results_df.to_csv(SEARCH_RESULTS_PATH, index=False, encoding=CSV_ENCODING)
    print(f"\nИтоговые результаты поиска ({len(search_results_df)} строк) сохранены: {SEARCH_RESULTS_PATH}")


if __name__ == "__main__":
    main()
