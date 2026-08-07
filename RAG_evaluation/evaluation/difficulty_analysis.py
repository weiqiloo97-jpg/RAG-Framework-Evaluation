"""
difficulty_analysis.py
======================
Difficulty Level Analysis module for Normal RAG evaluation.

Groups per-query Tier 1 and Tier 2 metric dicts by difficulty label
(easy / medium / hard) and reports averaged metrics per difficulty level.

This module is purely analytical — no retrieval or generation calls are
made here. It consumes the same output dicts produced by
``compute_tier1_metrics`` and ``compute_tier2_metrics``.

Public API
----------
    group_by_difficulty(queries, t1_runs, t2_runs=[])
        Group and average per-query metrics by difficulty label.

    print_tier1_difficulty_tables(all_pipeline_results)
        Print consolidated Tier 1 tables: one row per pipeline per difficulty.

    print_tier2_difficulty_table(pipeline_name, diff_results)
        Print Tier 2 table for a single representative pipeline.

    print_failure_diagnosis(pipeline_name, diff_results)
        Print objective failure diagnosis for Tier 2 results.

    write_tier1_csv(all_pipeline_results, path)
        Write Tier 1 difficulty results to CSV.

    write_tier2_csv(pipeline_name, diff_results, path)
        Write Tier 2 difficulty results to CSV.
"""

import csv
import numpy as np
from typing import List, Dict, Optional

# Canonical ordering for display
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

TIER1_KEYS = [
    "Hit Rate", "Precision@5", "Recall@5", "MRR",
    "Average Similarity", "Query Time",
]
TIER2_KEYS = [
    "Faithfulness", "Answer Relevancy", "Context Precision",
    "Context Recall", "Noise Sensitivity", "Robustness",
]


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def group_by_difficulty(
    queries: List[Dict],
    t1_runs: List[Dict],
    t2_runs: Optional[List[Dict]] = None,
) -> Dict[str, Dict]:
    """
    Groups per-query metric dicts by difficulty label and computes averages.

    Parameters
    ----------
    queries : list[dict]
        Query items from ``difficulty_queries.json``. Each must have a
        ``"difficulty"`` field (``"easy"``, ``"medium"``, or ``"hard"``).
    t1_runs : list[dict]
        Per-query Tier 1 metric dicts in the same order as ``queries``.
    t2_runs : list[dict], optional
        Per-query Tier 2 metric dicts in the same order as ``queries``.
        Pass an empty list or omit to skip Tier 2 averaging (e.g. for
        retrieval-only pipelines).

    Returns
    -------
    dict
        Keyed by ``"easy"``, ``"medium"``, ``"hard"``. Each value contains:
          - ``"count"``     : int
          - ``"tier1"``     : dict of averaged Tier 1 metrics
          - ``"tier2"``     : dict of averaged Tier 2 metrics (empty if not supplied)
          - ``"tier1_raw"`` : list[dict] per-query Tier 1 dicts
          - ``"tier2_raw"`` : list[dict] per-query Tier 2 dicts
    """
    if t2_runs is None:
        t2_runs = []

    buckets: Dict[str, Dict] = {
        level: {"t1_rows": [], "t2_rows": []}
        for level in DIFFICULTY_LEVELS
    }

    for idx, item in enumerate(queries):
        level = item.get("difficulty", "").lower().strip()
        if level not in buckets:
            level = "medium"
        if idx < len(t1_runs):
            buckets[level]["t1_rows"].append(t1_runs[idx])
        if idx < len(t2_runs):
            buckets[level]["t2_rows"].append(t2_runs[idx])

    results: Dict[str, Dict] = {}
    for level in DIFFICULTY_LEVELS:
        t1_rows = buckets[level]["t1_rows"]
        t2_rows = buckets[level]["t2_rows"]

        t1_avg: Dict[str, float] = {}
        for key in TIER1_KEYS:
            vals = [r[key] for r in t1_rows if key in r]
            t1_avg[key] = round(float(np.mean(vals)), 4) if vals else 0.0

        t2_avg: Dict[str, float] = {}
        for key in TIER2_KEYS:
            vals = [r[key] for r in t2_rows if key in r]
            t2_avg[key] = round(float(np.mean(vals)), 4) if vals else 0.0

        results[level] = {
            "count": len(t1_rows),
            "tier1": t1_avg,
            "tier2": t2_avg,
            "tier1_raw": t1_rows,
            "tier2_raw": t2_rows,
        }

    return results


# ---------------------------------------------------------------------------
# Tier 1 console output  (all pipelines, grouped by difficulty)
# ---------------------------------------------------------------------------

def print_tier1_difficulty_tables(
    all_pipeline_results: Dict[str, Dict[str, Dict]],
) -> None:
    """
    Prints a consolidated Tier 1 difficulty table.

    Layout: one difficulty block per section; within each block, one row
    per pipeline — making cross-pipeline comparisons easy to read.

    Parameters
    ----------
    all_pipeline_results : dict
        Keyed by pipeline name; values are outputs of ``group_by_difficulty()``.
    """
    sep = "=" * 90
    print(f"\n{sep}")
    print("   TIER 1 — RETRIEVAL DIFFICULTY ANALYSIS  (All Pipelines)")
    print(sep)

    col = (
        f"  {'Pipeline':<20} | {'Hit Rate':>9} | {'Prec@5':>7} | "
        f"{'Recall@5':>9} | {'MRR':>7} | {'Avg Sim':>8} | {'Time(s)':>8}"
    )

    for level in DIFFICULTY_LEVELS:
        # Count from first available pipeline
        count = 0
        for pr in all_pipeline_results.values():
            count = pr.get(level, {}).get("count", 0)
            if count:
                break

        print(f"\n  Difficulty: {level.upper()}  ({count} queries)")
        print("  " + "-" * 88)
        print(col)
        print("  " + "-" * 88)

        for pipeline_name, diff_results in all_pipeline_results.items():
            data = diff_results.get(level, {})
            t1 = data.get("tier1", {})
            print(
                f"  {pipeline_name:<20} | "
                f"{t1.get('Hit Rate', 0.0):>9.2%} | "
                f"{t1.get('Precision@5', 0.0):>7.4f} | "
                f"{t1.get('Recall@5', 0.0):>9.4f} | "
                f"{t1.get('MRR', 0.0):>7.4f} | "
                f"{t1.get('Average Similarity', 0.0):>8.4f} | "
                f"{t1.get('Query Time', 0.0):>8.4f}"
            )

    print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Tier 2 console output  (single representative pipeline)
# ---------------------------------------------------------------------------

def print_tier2_difficulty_table(
    pipeline_name: str,
    diff_results: Dict[str, Dict],
) -> None:
    """
    Prints the Tier 2 difficulty table for one representative pipeline.

    Parameters
    ----------
    pipeline_name : str
        E.g. ``"Hybrid + Rerank"``.
    diff_results : dict
        Output of ``group_by_difficulty()`` for this pipeline.
    """
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"   TIER 2 — GENERATION DIFFICULTY ANALYSIS  ({pipeline_name})")
    print(sep)

    header = (
        f"  {'Difficulty':<10} | {'Count':>5} | {'Faithful':>9} | "
        f"{'Relevancy':>9} | {'CtxPrec':>7} | {'CtxRec':>7} | "
        f"{'NoiseSens':>9} | {'Robust':>7}"
    )
    print(header)
    print("  " + "-" * 78)

    for level in DIFFICULTY_LEVELS:
        data = diff_results.get(level, {})
        t2 = data.get("tier2", {})
        print(
            f"  {level.capitalize():<10} | {data.get('count', 0):>5} | "
            f"{t2.get('Faithfulness', 0.0):>9.4f} | "
            f"{t2.get('Answer Relevancy', 0.0):>9.4f} | "
            f"{t2.get('Context Precision', 0.0):>7.4f} | "
            f"{t2.get('Context Recall', 0.0):>7.4f} | "
            f"{t2.get('Noise Sensitivity', 0.0):>9.4f} | "
            f"{t2.get('Robustness', 0.0):>7.4f}"
        )

    print("  " + "-" * 78)


# ---------------------------------------------------------------------------
# Failure diagnosis  (Tier 2 results, single pipeline)
# ---------------------------------------------------------------------------

def print_failure_diagnosis(
    pipeline_name: str,
    diff_results: Dict[str, Dict],
) -> None:
    """
    Prints an objective failure diagnosis derived from Tier 2 metrics.

    Reports the dominant bottleneck per difficulty level without assuming
    any expected direction of the trend.

    Parameters
    ----------
    pipeline_name : str
    diff_results : dict
        Output of ``group_by_difficulty()``.
    """
    print(f"\n  Failure Diagnosis — {pipeline_name}")
    print("  " + "-" * 72)

    for level in DIFFICULTY_LEVELS:
        data = diff_results.get(level, {})
        t1 = data.get("tier1", {})
        t2 = data.get("tier2", {})
        count = data.get("count", 0)

        if count == 0:
            print(f"  {level.capitalize():<8}: no queries in this level.")
            continue

        hit_rate = t1.get("Hit Rate", 0.0)
        faithfulness = t2.get("Faithfulness", 0.0)
        ctx_recall = t2.get("Context Recall", 0.0)

        retrieval_ok = hit_rate >= 0.5
        generation_ok = faithfulness >= 0.5

        if retrieval_ok and generation_ok:
            verdict = "Retrieval and generation both satisfactory."
        elif not retrieval_ok and generation_ok:
            verdict = "Primary bottleneck: retrieval (low Hit Rate)."
        elif retrieval_ok and not generation_ok:
            verdict = "Primary bottleneck: generation (correct retrieval, poor answer quality)."
        else:
            verdict = "Both retrieval and generation show weakness."

        print(
            f"  {level.capitalize():<8}: "
            f"Hit Rate={hit_rate:.2%}  "
            f"Faithfulness={faithfulness:.4f}  "
            f"CtxRecall={ctx_recall:.4f}  —  {verdict}"
        )

    print("  " + "-" * 72)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_tier1_csv(
    all_pipeline_results: Dict[str, Dict[str, Dict]],
    path: str,
) -> None:
    """
    Writes Tier 1 difficulty results for all pipelines to CSV.

    Parameters
    ----------
    all_pipeline_results : dict
        Keyed by pipeline name; values are outputs of ``group_by_difficulty()``.
    path : str
        Output CSV file path.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Difficulty", "Count",
            "Hit Rate", "Precision@5", "Recall@5", "MRR",
            "Average Similarity", "Avg Query Time (s)",
        ])
        for pipeline_name, diff_result in all_pipeline_results.items():
            for level in DIFFICULTY_LEVELS:
                data = diff_result.get(level, {})
                t1 = data.get("tier1", {})
                writer.writerow([
                    pipeline_name,
                    level.capitalize(),
                    data.get("count", 0),
                    f"{t1.get('Hit Rate', 0.0):.4f}",
                    f"{t1.get('Precision@5', 0.0):.4f}",
                    f"{t1.get('Recall@5', 0.0):.4f}",
                    f"{t1.get('MRR', 0.0):.4f}",
                    f"{t1.get('Average Similarity', 0.0):.4f}",
                    f"{t1.get('Query Time', 0.0):.4f}",
                ])


def write_tier2_csv(
    pipeline_name: str,
    diff_results: Dict[str, Dict],
    path: str,
) -> None:
    """
    Writes Tier 2 difficulty results for a single pipeline to CSV.

    Parameters
    ----------
    pipeline_name : str
    diff_results : dict
        Output of ``group_by_difficulty()``.
    path : str
        Output CSV file path.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Difficulty", "Count",
            "Faithfulness", "Answer Relevancy", "Context Precision",
            "Context Recall", "Noise Sensitivity", "Robustness",
        ])
        for level in DIFFICULTY_LEVELS:
            data = diff_results.get(level, {})
            t2 = data.get("tier2", {})
            writer.writerow([
                pipeline_name,
                level.capitalize(),
                data.get("count", 0),
                f"{t2.get('Faithfulness', 0.0):.4f}",
                f"{t2.get('Answer Relevancy', 0.0):.4f}",
                f"{t2.get('Context Precision', 0.0):.4f}",
                f"{t2.get('Context Recall', 0.0):.4f}",
                f"{t2.get('Noise Sensitivity', 0.0):.4f}",
                f"{t2.get('Robustness', 0.0):.4f}",
            ])
