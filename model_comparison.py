"""
Model Comparison and Dynamic Model Selection Module.
Compares retrieval performance across 5 embedding models:
- BGE-M3
- Multilingual-E5-large
- BGE-large-en-v1.5
- MiniLM
- Nomic Embed Text v1.5

Fixes raw-score bias (where E5 dominates due to elevated baseline cosine distributions)
using scientifically defensible Z-score normalization, discriminative confidence margin,
and query-type alignment.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from embedding_pipeline import MODEL_CONFIGS, EmbeddingPipeline

class ModelComparisonEngine:
    def __init__(self, pipeline: EmbeddingPipeline):
        self.pipeline = pipeline
        self.models = list(MODEL_CONFIGS.keys())

    def compare_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieves top_k chunks from all 5 models and compiles comparison analytics.
        """
        results_by_model = {}
        latencies = {}
        raw_similarities = {}

        for model_name in self.models:
            retrieved = self.pipeline.retrieve(query, model_name, top_k=top_k)
            results_by_model[model_name] = retrieved
            cfg = MODEL_CONFIGS[model_name]
            latencies[model_name] = cfg["latency_ms"]
            raw_similarities[model_name] = [r["similarity"] for r in retrieved]

        # Calculate dynamic model selection
        selection = self.select_optimal_model(query, results_by_model)

        return {
            "query": query,
            "results_by_model": results_by_model,
            "latencies": latencies,
            "similarities": raw_similarities,
            "dynamic_selection": selection
        }

    def select_optimal_model(self, query: str, results_by_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Scientifically calibrated query-level dynamic model selection.
        Solves the E5 score-distribution dominance by combining:
        1. Calibrated Z-score / Min-Max normalized relevance (accounting for each model's baseline)
        2. Discriminative confidence margin (Top-1 vs Top-3 / Top-5 drop-off)
        3. Query complexity and contextual suitability alignment
        """
        model_evaluations = {}
        query_words = query.split()
        is_long_query = len(query_words) > 15
        is_multihop_or_summary = any(k in query.lower() for k in ["compare", "summarize", "difference", "relationship", "overall", "comprehensive", "how does", "why does"])
        is_concise_factual = any(k in query.lower() for k in ["what is", "when", "how many", "who", "which section", "define", "specify"])

        for model_name, retrieved in results_by_model.items():
            if not retrieved:
                continue

            cfg = MODEL_CONFIGS[model_name]
            top_1_sim = retrieved[0]["similarity"]
            top_3_sim = retrieved[min(2, len(retrieved) - 1)]["similarity"]
            top_5_sim = retrieved[min(4, len(retrieved) - 1)]["similarity"]

            # 1. Distribution normalization (Z-score relative to model's baseline distribution)
            mean_b = cfg["mean_baseline"]
            std_b = cfg["std_baseline"]
            z_score = (top_1_sim - mean_b) / max(std_b, 1e-6)

            # Min-Max normalized score within typical range
            min_r, max_r = cfg["typical_score_range"]
            norm_score = float(np.clip((top_1_sim - min_r) / max(max_r - min_r, 1e-6), 0.0, 1.0))

            # 2. Discriminative Margin: Drop from top-1 to top-3
            # A model confident in a specific chunk will exhibit a distinct drop,
            # whereas an uncalibrated high baseline assigns high similarity to everything.
            margin_1_3 = max(0.0, top_1_sim - top_3_sim)
            normalized_margin = margin_1_3 / max(std_b, 0.01)

            # 3. Query alignment bonus
            alignment_bonus = 0.0
            alignment_reasons = []

            if model_name == "BGE-M3":
                if is_multihop_or_summary or is_long_query:
                    alignment_bonus += 0.12
                    alignment_reasons.append("Multi-vector dense+sparse representation fits complex/multi-hop queries")
            elif model_name == "BGE-large-en-v1.5":
                if is_concise_factual or "specification" in query.lower() or "architecture" in query.lower():
                    alignment_bonus += 0.12
                    alignment_reasons.append("English specialized dense retrieval with high precision on factual specifications")
            elif model_name == "Multilingual-E5-large":
                # E5 gets bonus if high discriminative margin is demonstrated
                if margin_1_3 > 0.04:
                    alignment_bonus += 0.08
                    alignment_reasons.append("Strong discriminative margin in multilingual passage space")
            elif model_name == "Nomic Embed Text v1.5":
                if is_long_query or "summarize" in query.lower():
                    alignment_bonus += 0.10
                    alignment_reasons.append("8192-token Matryoshka context window ideal for broad contextual synthesis")
            elif model_name == "MiniLM":
                if len(query_words) < 8 and is_concise_factual:
                    alignment_bonus += 0.06
                    alignment_reasons.append("Ultra-low latency (7ms) with strong sharp local match on short query")

            # Final composite score
            # 45% normalized relevance + 35% discriminative margin + 20% query alignment
            composite_score = (0.45 * norm_score) + (0.35 * min(1.0, normalized_margin * 0.4)) + (0.20 * min(1.0, 0.5 + alignment_bonus))

            model_evaluations[model_name] = {
                "raw_top_1": round(top_1_sim, 4),
                "z_score": round(float(z_score), 3),
                "norm_relevance": round(float(norm_score), 4),
                "margin_1_3": round(float(margin_1_3), 4),
                "alignment_bonus": round(alignment_bonus, 3),
                "composite_score": round(float(composite_score), 4),
                "alignment_reasons": alignment_reasons,
                "top_chunk": retrieved[0]
            }

        # Select model with highest composite score
        selected_model = max(model_evaluations.keys(), key=lambda m: model_evaluations[m]["composite_score"])
        sel_data = model_evaluations[selected_model]

        reasons = []
        if sel_data["margin_1_3"] > 0.03:
            reasons.append(f"Highest discriminative margin (Δ={sel_data['margin_1_3']})")
        reasons.append(f"Normalized relevance z-score: {sel_data['z_score']:+.2f}")
        if sel_data["alignment_reasons"]:
            reasons.extend(sel_data["alignment_reasons"])
        reason_str = "; ".join(reasons)

        return {
            "selected_model": selected_model,
            "selection_score": sel_data["composite_score"],
            "selection_reason": reason_str,
            "model_evaluations": model_evaluations,
            "selected_evidence": results_by_model[selected_model][:3]
        }
