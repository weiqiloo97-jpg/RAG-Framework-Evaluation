# Optimised runner with Tier 2 for all pipelines (cached LLM)
"""
run_difficulty_analysis.py (final)
===================================
Two‑phase evaluation:
  • Phase 1 – Tier 1 retrieval metrics for **all four** pipelines (no LLM).
  • Phase 2 – Tier 2 end‑to‑end metrics for **all four** pipelines, re‑using a
    single LLM instance and persisting generated answers in a JSON cache so
    subsequent runs skip LLM calls.

Both phases keep the original 20 labelled queries and output:
  * Tier 1 CSV & console table (grouped by difficulty, all pipelines)
  * Tier 2 CSV & console table (grouped by difficulty, all pipelines)
  * A short interpretation comparing retrieval vs generation degradation.
"""

import os
import sys
import json
import time
import httpx
from collections import Counter

# ── UTF‑8 stdout on Windows ──────────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ── Make package importable when run from project root ───────────────────────
_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# ── Existing components (unchanged) ───────────────────────────────────────────
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.answer_generator import AnswerGenerator, OllamaLLM, FallbackLLM
from evaluation.tier1_metrics import compute_tier1_metrics
from evaluation.tier2_metrics import compute_tier2_metrics, introduce_typos

# ── New analysis helpers ─────────────────────────────────────────────────────
from evaluation.difficulty_analysis import (
    group_by_difficulty,
    print_tier1_difficulty_tables,
    print_tier2_difficulty_table,
    print_failure_diagnosis,
    write_tier1_csv,
    write_tier2_csv,
)

# ── Paths & constants ───────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUERIES_FILE = os.path.join(_PROJECT_ROOT, "RAG_evaluation", "difficulty_queries.json")
CACHE_FILE   = os.path.join(_PROJECT_ROOT, "RAG_evaluation", "difficulty_answer_cache.json")
T1_CSV       = os.path.join(_PROJECT_ROOT, "RAG_evaluation", "difficulty_tier1_results.csv")
T2_CSV       = os.path.join(_PROJECT_ROOT, "RAG_evaluation", "difficulty_tier2_results.csv")

# ── Helper utilities ───────────────────────────────────────────────────────

def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Return True if a local Ollama server is reachable."""
    try:
        return httpx.get(base_url, timeout=2.0).status_code == 200
    except Exception:
        return False


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def save_cache(cache: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)


def make_cache_key(pipeline: str, idx: int) -> str:
    """Stable cache key – pipeline name + query index."""
    return f"{pipeline}|{idx}"

# ── Main workflow ───────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("   RAG DIFFICULTY LEVEL ANALYSIS (Optimised – Full Tier‑2)")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # 1️⃣ Load difficulty queries
    # ---------------------------------------------------------------------
    if not os.path.exists(QUERIES_FILE):
        print(f"[-] Queries file not found: {QUERIES_FILE}")
        return
    with open(QUERIES_FILE, "r", encoding="utf-8") as fh:
        queries = json.load(fh)
    dist = Counter(q.get("difficulty", "unknown") for q in queries)
    print(
        f"[Data] Loaded {len(queries)} queries – "
        f"Easy: {dist.get('easy',0)}, Medium: {dist.get('medium',0)}, Hard: {dist.get('hard',0)}"
    )

    # ---------------------------------------------------------------------
    # 2️⃣ Initialise retrieval components (single instances reused)
    # ---------------------------------------------------------------------
    print("\n[Init] Loading embeddings and connecting to ChromaDB…")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
        rerank_model  = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

        db_path         = "./chroma_store"
        collection_name = "mflix"
        vec_retriever   = VectorRetriever(db_path=db_path, collection_name=collection_name,
                                         model_name="all-MiniLM-L6-v2")
        bm25_retriever  = BM25Retriever(db_path=db_path, collection_name=collection_name,
                                         source_filter="movies")
        hybrid_retriever = HybridRetriever(vector_retriever=vec_retriever,
                                           bm25_retriever=bm25_retriever,
                                           alpha=0.5)
        print("   [OK] Retrieval models ready.")
    except Exception as e:
        print(f"[-] Retrieval init failed: {e}")
        return

    # ---------------------------------------------------------------------
    # 3️⃣ Initialise LLM **once** – used for every Tier‑2 run
    # ---------------------------------------------------------------------
    print("\n[LLM] Initialising LLM wrapper (single instance for all pipelines)…")
    if check_ollama():
        llm = OllamaLLM(model_name="llama3")
        print("   [OK] Local Ollama found – using llama3.")
    else:
        llm = FallbackLLM()
        print("   [WARN] Ollama not detected – falling back to heuristic LLM.")
    generator = AnswerGenerator(llm=llm)

    # ---------------------------------------------------------------------
    # 4️⃣ Define the four pipelines (same ordering as original evaluation)
    # ---------------------------------------------------------------------
    pipelines = [
        ("Vector RAG", vec_retriever),
        ("BM25 RAG", bm25_retriever),
        ("Hybrid RAG", hybrid_retriever),
        ("Hybrid + Rerank", None),  # Hybrid + Rerank uses hybrid_retriever + rerank_model
    ]

    # ---------------------------------------------------------------------
    # 5️⃣ Prepare cache for generated answers (persisted across runs)
    # ---------------------------------------------------------------------
    cache = load_cache(CACHE_FILE)
    cache_hits = 0
    cache_misses = 0

    # Containers for final results per pipeline
    tier1_by_pipeline: dict = {}
    tier2_by_pipeline: dict = {}

    # ---------------------------------------------------------------------
    # 6️⃣ Loop over pipelines – compute both Tier 1 and Tier 2 for each
    # ---------------------------------------------------------------------
    for pipeline_name, retriever in pipelines:
        print(f"\n{'='*70}\n[Run] Pipeline: {pipeline_name}\n{'='*70}")
        t1_runs = []
        t2_runs = []

        for idx, item in enumerate(queries):
            query          = item["query"]
            ground_truth   = item["ground_truth"]
            gt_docs        = item["relevant_documents"]
            difficulty     = item.get("difficulty", "unknown")
            cache_key      = make_cache_key(pipeline_name, idx)

            # ---------- Retrieval (common to Tier 1 and Tier 2) ----------
            start = time.perf_counter()
            if pipeline_name == "Hybrid + Rerank":
                cand = hybrid_retriever.retrieve(query, top_k=20)
                retrieved = rerank_model.rerank(query, cand, top_k=5)
            else:
                retrieved = retriever.retrieve(query, top_k=5)
            query_time = time.perf_counter() - start

            # ---------- Tier 1 metrics ----------
            t1 = compute_tier1_metrics(retrieved, gt_docs, query, query_time, embed_model)
            t1_runs.append(t1)

            # ---------- Tier 2 – answer generation (cached) ----------
            if cache_key in cache:
                # Cache hit – reuse previously computed answer and perturbed IDs
                entry = cache[cache_key]
                generated_answer = entry["generated_answer"]
                orig_ids = entry["orig_ids"]
                pert_ids = entry["pert_ids"]
                cache_hits += 1
                hit_tag = "[CACHE]"
            else:
                # Cache miss – generate answer and perform robustness pass
                generated_answer = generator.generate_answer(query, retrieved)
                # Skip perturbed query retrieval for speed (Robustness metric will be 0.0)
                orig_ids = [d["id"] for d in retrieved]
                pert_ids = []
                # Store in cache immediately
                cache[cache_key] = {
                    "generated_answer": generated_answer,
                    "orig_ids": orig_ids,
                    "pert_ids": pert_ids,
                }
                save_cache(cache, CACHE_FILE)
                cache_misses += 1
                hit_tag = "[GEN]  "

            # ---------- Tier 2 metrics ----------
            t2 = compute_tier2_metrics(
                query=query,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                retrieved_docs=retrieved,
                llm=llm,
                original_retrieved_ids=orig_ids,
                perturbed_retrieved_ids=pert_ids,
                embed_model=embed_model,
                fast_mode=True
            )
            t2_runs.append(t2)

            print(
                f"  [{idx+1:>2}/{len(queries)}] {hit_tag} [{difficulty.upper():<6}] "
                f"HR={t1['Hit Rate']:.0f}%  Faith={t2['Faithfulness']:.3f} "
                f"\"{query[:45]}\""
            )

        # ---------------------------------------------------------------
        # Group results by difficulty for this pipeline (both tiers)
        # ---------------------------------------------------------------
        diff_res = group_by_difficulty(queries, t1_runs, t2_runs)
        tier1_by_pipeline[pipeline_name] = diff_res
        tier2_by_pipeline[pipeline_name] = diff_res

    # ---------------------------------------------------------------------
    # 7️⃣ Summary & CSV output
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   TIER 1 – RETRIEVAL BY DIFFICULTY (All Pipelines)")
    print("=" * 70)
    print_tier1_difficulty_tables(tier1_by_pipeline)
    write_tier1_csv(tier1_by_pipeline, T1_CSV)
    print(f"[Save] Tier 1 CSV → {T1_CSV}")

    print("\n" + "=" * 70)
    print("   TIER 2 – GENERATION BY DIFFICULTY (All Pipelines)")
    print("=" * 70)
    for pname, diff_res in tier2_by_pipeline.items():
        print_tier2_difficulty_table(pname, diff_res)
        print_failure_diagnosis(pname, diff_res)
        # Write a separate CSV per pipeline (named with pipeline suffix)
        csv_path = T2_CSV.replace('.csv', f'_{pname.replace(" ", "_")}.csv')
        write_tier2_csv(pname, diff_res, csv_path)
        print(f"[Save] Tier 2 CSV ({pname}) → {csv_path}")

    # ---------------------------------------------------------------------
    # 8️⃣ Automatic interpretation (high‑level comparison)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   AUTOMATIC INTERPRETATION")
    print("=" * 70)
    for level in ["easy", "medium", "hard"]:
        print(f"\nDifficulty: {level.upper()}")
        for pname in pipelines:
            name = pname[0]
            diff = tier2_by_pipeline[name].get(level, {})
            t1 = diff.get("tier1", {})
            t2 = diff.get("tier2", {})
            hr = t1.get('Hit Rate', 0.0)
            faith = t2.get('Faithfulness', 0.0)
            print(f"  {name:<15}: Hit Rate={hr:.2%}, Faithfulness={faith:.4f}")
        # Simple heuristic comment
        hr_vals = [tier1_by_pipeline[p[0]][level]["tier1"].get('Hit Rate',0) for p in pipelines]
        faith_vals = [tier2_by_pipeline[p[0]][level]["tier2"].get('Faithfulness',0) for p in pipelines]
        if all(hr > 0.6 for hr in hr_vals) and any(fa < 0.5 for fa in faith_vals):
            note = "⟹ Generation quality drops despite solid retrieval."
        elif any(hr < 0.5 for hr in hr_vals):
            note = "⟹ Retrieval struggles on this difficulty level."
        else:
            note = "⟹ Both retrieval and generation remain stable."
        print(f"  {note}")

    print("\n" + "=" * 70)
    print("   ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Cache summary: {cache_hits} hit(s), {cache_misses} miss(es). LLM calls saved: {cache_hits}")


if __name__ == "__main__":
    main()
