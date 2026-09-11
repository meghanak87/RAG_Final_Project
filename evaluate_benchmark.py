"""
Benchmark Evaluation Module.
Processes raw benchmark_results.json and computes:
- Retrieval metrics (Recall@1, Recall@3, Recall@5, MRR) for all 5 models + Dynamic Selection
- Generation metrics (Correctness, Relevance, Grounding)
- Unanswerable detection accuracy & false answer count
- Category-wise breakdowns (factual, conceptual, comparison, multi-hop, summarization, unanswerable)
- Saves evaluation_metrics.json and evaluation_report.json
"""

import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation_engine import EvaluationEngine

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
METRICS_FILE = os.path.join(os.path.dirname(__file__), "evaluation_metrics.json")
REPORT_FILE = os.path.join(os.path.dirname(__file__), "evaluation_report.json")

def evaluate_all(results=None):
    if results is None:
        if not os.path.exists(RESULTS_FILE):
            from benchmark_runner import run_benchmark
            results = run_benchmark()
        else:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)

    engine = EvaluationEngine()
    models = ["BGE-M3", "Multilingual-E5-large", "BGE-large-en-v1.5", "MiniLM", "Nomic Embed Text v1.5"]

    print("Evaluating 5 models + Dynamic Selection across all 50 questions...")

    # 1. Evaluate each individual model
    model_metrics = {}
    for m in models:
        q_evals = []
        for item in results:
            retrieved_chunks = item["models_retrieval"][m]["top_5"]
            retrieval_res = engine.evaluate_retrieval_for_item(retrieved_chunks, item)
            
            # Answer eval: compare top chunk excerpt against ground truth
            top_text = retrieved_chunks[0]["text"] if retrieved_chunks else ""
            ans_eval = engine.evaluate_answer_correctness(top_text, item["ground_truth"], item["category"])
            grounding_eval = engine.evaluate_grounding(top_text, retrieved_chunks, item["expected_evidence"], item["category"])
            
            q_evals.append({
                "id": item["id"],
                "category": item["category"],
                "retrieval": retrieval_res,
                "answer_eval": ans_eval,
                "grounding": grounding_eval,
                "latency_ms": item["model_latencies"].get(m, 25.0)
            })
            
        model_metrics[m] = engine.aggregate_metrics(q_evals)

    # 2. Evaluate Dynamic Selection
    dynamic_evals = []
    selection_counts = {m: 0 for m in models}

    for item in results:
        sel_m = item["selected_model"]
        selection_counts[sel_m] = selection_counts.get(sel_m, 0) + 1
        
        # Use selected evidence
        sel_chunks = item["selected_evidence"]
        retrieval_res = engine.evaluate_retrieval_for_item(sel_chunks, item)
        ans_eval = engine.evaluate_answer_correctness(item["generated_answer"], item["ground_truth"], item["category"])
        grounding_eval = engine.evaluate_grounding(item["generated_answer"], sel_chunks, item["expected_evidence"], item["category"])

        dynamic_evals.append({
            "id": item["id"],
            "category": item["category"],
            "retrieval": retrieval_res,
            "answer_eval": ans_eval,
            "grounding": grounding_eval,
            "latency_ms": item.get("total_latency_ms", 35.0),
            "selected_model": sel_m,
            "selection_score": item.get("selection_score", 0.0),
            "selection_reason": item.get("selection_reason", "")
        })

    dynamic_summary = engine.aggregate_metrics(dynamic_evals)
    dynamic_summary["selection_distribution"] = {
        m: {
            "count": cnt,
            "percentage": round(cnt / len(results) * 100, 1)
        }
        for m, cnt in selection_counts.items()
    }

    # 3. Compile full evaluation report
    report = {
        "benchmark_summary": {
            "total_questions": len(results),
            "categories": ["factual", "conceptual", "comparison", "multi-hop", "summarization", "unanswerable"],
            "models_evaluated": models + ["Dynamic Selection"],
            "unbiased_selection_status": "Active (Z-score + Discriminative Margin + Query Alignment)"
        },
        "dynamic_selection": dynamic_summary,
        "models": model_metrics,
        "category_breakdown": dynamic_summary["by_category"],
        "unanswerable_evaluation": {
            "total_unanswerable": len([q for q in results if q["category"] == "unanswerable"]),
            "unanswerable_detection_accuracy": dynamic_summary["unanswerable_detection_accuracy"],
            "false_answer_count": dynamic_summary["false_answer_count"],
            "status": "PASSED - No false answers generated" if dynamic_summary["false_answer_count"] == 0 else "REVIEW"
        }
    }

    # Save evaluation_metrics.json
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save evaluation_report.json (with question-level details)
    full_report = dict(report)
    full_report["question_details"] = dynamic_evals
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Metrics saved to {METRICS_FILE}")
    print(f"Full report saved to {REPORT_FILE}")

    return report

if __name__ == "__main__":
    evaluate_all()
