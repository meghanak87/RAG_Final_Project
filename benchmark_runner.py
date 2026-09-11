"""
Benchmark Runner Module.
Executes the full 50-question benchmark through the real RAG pipeline:
1. Ingests source document via PDFProcessor & Chunker.
2. Embeds and indexes chunks across all 5 models via FaissVectorIndex.
3. For each of the 50 questions:
   - Retrieves Top-1, Top-3, Top-5 from:
     * BGE-M3
     * Multilingual-E5-large
     * BGE-large-en-v1.5
     * MiniLM
     * Nomic Embed Text v1.5
   - Executes unbiased Query-level Dynamic Model Selection.
   - Generates grounded answer from selected evidence.
4. Saves raw results to evaluation/benchmark_results.json.
"""

import os
import json
import time
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_engine import RAGEngine
from pdf_processor import PDFProcessor
from chunker import DocumentChunker

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "benchmark.json")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
SOURCE_DOC_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "source_document.txt")

def run_benchmark(limit: int = None, verbose: bool = True) -> list:
    print("=" * 60)
    print("STARTING RAG BENCHMARK RUNNER (50 Questions)")
    print("=" * 60)

    # 1. Initialize RAG Engine (using grounded synthesis for fast, reliable, reproducible evaluation)
    rag = RAGEngine(gemini_api_key="")

    # 2. Ingest source document
    if not os.path.exists(SOURCE_DOC_FILE):
        raise FileNotFoundError(f"Source document not found at {SOURCE_DOC_FILE}")

    with open(SOURCE_DOC_FILE, "rb") as f:
        file_bytes = f.read()

    filename = "Enterprise_AI_Architecture_and_Governance_Specification_2025.pdf"
    print(f"Ingesting source document: {filename} ({len(file_bytes)} bytes)...")
    ingest_info = rag.ingest_document(file_bytes, filename)
    print(f"Ingestion complete: {ingest_info['total_pages']} pages, {ingest_info['total_chunks']} chunks created.")

    # 3. Load benchmark dataset
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    if limit:
        benchmark_items = benchmark_items[:limit]

    total_q = len(benchmark_items)
    print(f"Loaded {total_q} benchmark items from {BENCHMARK_FILE}")

    results = []
    models = ["BGE-M3", "Multilingual-E5-large", "BGE-large-en-v1.5", "MiniLM", "Nomic Embed Text v1.5"]

    start_benchmark_time = time.time()

    for idx, item in enumerate(benchmark_items, start=1):
        q_id = item["id"]
        question = item["question"]
        category = item["category"]

        if verbose:
            print(f"[{idx}/{total_q}] ({category}) {q_id}: {question[:65]}...")

        # Run query through real pipeline (Retrieval across 5 models + Dynamic Selection + Answer Generation)
        q_start = time.time()
        rag_res = rag.query(question, top_k=5)
        total_q_time = (time.time() - q_start) * 1000.0

        # Construct benchmark result record
        record = {
            "id": q_id,
            "question": question,
            "category": category,
            "ground_truth": item["ground_truth"],
            "expected_evidence": item["expected_evidence"],
            "source": item["source"],
            "page": item["page"],
            "selected_model": rag_res["selected_model"],
            "selection_score": rag_res["selection_score"],
            "selection_reason": rag_res["selection_reason"],
            "selected_evidence": rag_res["selected_evidence"],
            "generated_answer": rag_res["generated_answer"],
            "grounding_score": rag_res["grounding_score"],
            "grounding_status": rag_res["grounding_status"],
            "is_unanswerable": rag_res["is_unanswerable"],
            "total_latency_ms": round(total_q_time, 2),
            "model_latencies": rag_res["latencies"],
            # Top-1, Top-3, Top-5 retrieval results for EVERY model
            "models_retrieval": {
                m: {
                    "top_1": rag_res["results_by_model"][m][:1],
                    "top_3": rag_res["results_by_model"][m][:3],
                    "top_5": rag_res["results_by_model"][m][:5]
                }
                for m in models
            }
        }
        results.append(record)

    # Save to benchmark_results.json
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_benchmark_time
    print("=" * 60)
    print(f"BENCHMARK COMPLETED in {elapsed:.2f}s! Saved {len(results)} items to {RESULTS_FILE}")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_benchmark()
