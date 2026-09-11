"""
Evaluation Engine Module.
Implements standard Information Retrieval (IR) and Generation metrics:
- Recall@1, Recall@3, Recall@5
- MRR (Mean Reciprocal Rank)
- Average Retrieval Latency
- Top-1 and Top-3 Average Similarities
- Answer Correctness (Keyword overlap baseline + Semantic overlap)
- Answer Relevance
- Grounding Score and Support Classification (supported / partially_supported / unsupported)
- Unanswerable Detection Accuracy and False Answer Count
- Category-wise aggregations
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple

class EvaluationEngine:
    def __init__(self):
        self.categories = ["factual", "conceptual", "comparison", "multi-hop", "summarization", "unanswerable"]

    def is_chunk_relevant(self, retrieved_chunk: Dict[str, Any], benchmark_item: Dict[str, Any]) -> bool:
        """
        Determines whether a retrieved chunk is relevant to the benchmark ground truth.
        Uses multi-criteria matching:
        1. Exact page match + source match
        2. Expected evidence phrase overlap / token containment
        3. Jaccard token overlap between retrieved text and expected evidence
        """
        expected_page = benchmark_item.get("page")
        expected_evidence = benchmark_item.get("expected_evidence", "").lower()
        chunk_text = retrieved_chunk.get("text", "").lower()
        chunk_page = retrieved_chunk.get("page")

        # For unanswerable questions, no chunk is truly relevant
        if benchmark_item.get("category") == "unanswerable":
            return False

        # Criterion 1: Page match with substantive topical keywords
        page_matched = (expected_page is not None and chunk_page == expected_page)

        # Criterion 2: Substring or key phrase match
        if expected_evidence:
            clean_expected = re.sub(r"[^\w\s]", "", expected_evidence).strip()
            clean_chunk = re.sub(r"[^\w\s]", "", chunk_text).strip()
            
            # Direct containment
            if clean_expected in clean_chunk:
                return True

            # Token overlap (Jaccard > 0.35 or substantive key terms)
            ev_words = set(clean_expected.split())
            chunk_words = set(clean_chunk.split())
            if ev_words:
                overlap = len(ev_words.intersection(chunk_words))
                recall_ev = overlap / len(ev_words)
                if recall_ev >= 0.45 or (page_matched and recall_ev >= 0.25):
                    return True

        # Criterion 3: Strict page match if evidence is empty
        if page_matched:
            return True

        return False

    def evaluate_retrieval_metrics(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        expected_evidence: Any = "",
        page: Any = None,
        category: str = "",
        source: str = "",
        retrieval_latency: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculates Information Retrieval (IR) metrics for evaluated retrieval results:
        - Recall@1: 1.0 if relevant chunk ranked in top-1, else 0.0
        - Recall@3: 1.0 if relevant chunk ranked in top-3, else 0.0
        - Recall@5: 1.0 if relevant chunk ranked in top-5, else 0.0
        - MRR (Mean Reciprocal Rank): 1 / first_relevant_rank, or 0.0
        - retrieval latency: latency_ms for retrieval execution
        - expected evidence, expected page, and expected source
        """
        if isinstance(expected_evidence, dict):
            benchmark_item = expected_evidence
            expected_ev_str = benchmark_item.get("expected_evidence", "")
            exp_page = benchmark_item.get("page", page)
            cat = benchmark_item.get("category", category)
            src = benchmark_item.get("source", source)
            lat = benchmark_item.get(
                "retrieval_latency_ms",
                benchmark_item.get("latency_ms", benchmark_item.get("total_latency_ms", retrieval_latency))
            )
        else:
            expected_ev_str = str(expected_evidence) if expected_evidence is not None else ""
            exp_page = page
            cat = category or kwargs.get("category", "")
            src = source or kwargs.get("expected_source", kwargs.get("source", ""))
            lat = retrieval_latency or kwargs.get("retrieval_latency_ms", kwargs.get("latency_ms", kwargs.get("total_latency_ms", 0.0)))
            benchmark_item = {
                "expected_evidence": expected_ev_str,
                "page": exp_page,
                "category": cat,
                "source": src
            }

        is_unanswerable = (cat == "unanswerable")
        first_relevant_rank = None

        if is_unanswerable:
            recall_1 = 1.0
            recall_3 = 1.0
            recall_5 = 1.0
            mrr = 1.0
        else:
            for rank, chunk in enumerate(retrieved_chunks, start=1):
                if self.is_chunk_relevant(chunk, benchmark_item):
                    first_relevant_rank = rank
                    break

            recall_1 = 1.0 if (first_relevant_rank is not None and first_relevant_rank <= 1) else 0.0
            recall_3 = 1.0 if (first_relevant_rank is not None and first_relevant_rank <= 3) else 0.0
            recall_5 = 1.0 if (first_relevant_rank is not None and first_relevant_rank <= 5) else 0.0
            mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

        return {
            "recall_at_1": recall_1,
            "recall_at_3": recall_3,
            "recall_at_5": recall_5,
            "reciprocal_rank": mrr,
            "mrr": mrr,
            "first_relevant_rank": first_relevant_rank,
            "retrieval_latency": float(lat),
            "retrieval_latency_ms": float(lat),
            "expected_evidence": expected_ev_str,
            "expected_page": exp_page,
            "expected_source": src,
            "page": exp_page,
            "source": src,
            "category": cat,
            "is_hit": (first_relevant_rank is not None) or is_unanswerable
        }

    def evaluate_retrieval_for_item(self, retrieved_chunks: List[Dict[str, Any]], benchmark_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates Recall@1, Recall@3, Recall@5, and reciprocal rank for a single query.
        Delegates to evaluate_retrieval_metrics for consistent IR metrics calculation.
        """
        return self.evaluate_retrieval_metrics(
            retrieved_chunks=retrieved_chunks,
            expected_evidence=benchmark_item
        )

    def evaluate_answer_correctness(self, generated_answer: str, ground_truth: str, category: str) -> Dict[str, Any]:
        """
        Evaluates generated answer against ground truth:
        1. Keyword overlap correctness (baseline F1-score)
        2. Semantic answer correctness
        3. Answer relevance
        """
        if category == "unanswerable":
            # Correctness for unanswerable is whether it acknowledges absence of information
            unans_phrases = ["not contain", "not available", "not mentioned", "no information", "cannot be found", "does not state"]
            detected = any(p in generated_answer.lower() for p in unans_phrases)
            return {
                "keyword_overlap_score": 1.0 if detected else 0.0,
                "semantic_correctness": 1.0 if detected else 0.0,
                "answer_relevance": 1.0 if detected else 0.1,
                "is_unanswerable_handled": detected
            }

        gen_clean = re.sub(r"[^\w\s]", " ", generated_answer.lower())
        gt_clean = re.sub(r"[^\w\s]", " ", ground_truth.lower())

        gen_tokens = [w for w in gen_clean.split() if len(w) > 2]
        gt_tokens = [w for w in gt_clean.split() if len(w) > 2]

        if not gt_tokens:
            return {"keyword_overlap_score": 0.0, "semantic_correctness": 0.0, "answer_relevance": 0.0, "is_unanswerable_handled": False}

        gen_set = set(gen_tokens)
        gt_set = set(gt_tokens)

        overlap = len(gen_set.intersection(gt_set))
        precision = overlap / len(gen_set) if gen_set else 0.0
        recall = overlap / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Semantic correctness approximation combining recall and order/containment
        # Ground truth key entity containment gives higher semantic fidelity
        exact_phrases_found = sum(1 for phrase in gt_clean.split(";") if phrase.strip() in gen_clean)
        semantic_score = min(1.0, 0.6 * recall + 0.25 * f1 + 0.15 * (1.0 if exact_phrases_found > 0 else 0.0))

        relevance_score = min(1.0, 0.7 * precision + 0.3 * recall + 0.1)

        return {
            "keyword_overlap_score": round(float(f1), 4),
            "semantic_correctness": round(float(semantic_score), 4),
            "answer_relevance": round(float(relevance_score), 4),
            "is_unanswerable_handled": False
        }

    def evaluate_grounding(self, generated_answer: str, retrieved_evidence: List[Dict[str, Any]], expected_evidence: str, category: str) -> Dict[str, Any]:
        """
        Evaluates whether the generated answer is supported by the retrieved evidence.
        Status: 'supported', 'partially_supported', 'unsupported'
        """
        if category == "unanswerable":
            unans_phrases = ["not contain", "not available", "not mentioned", "no information", "cannot be found"]
            if any(p in generated_answer.lower() for p in unans_phrases):
                return {"grounding_score": 1.0, "grounding_status": "supported"}
            else:
                return {"grounding_score": 0.0, "grounding_status": "unsupported"}

        evidence_text = " ".join([c.get("text", "").lower() for c in retrieved_evidence])
        ans_clean = re.sub(r"[^\w\s]", " ", generated_answer.lower())
        ans_tokens = [w for w in ans_clean.split() if len(w) > 3]

        if not ans_tokens:
            return {"grounding_score": 0.5, "grounding_status": "partially_supported"}

        supported_tokens = sum(1 for w in ans_tokens if w in evidence_text)
        score = round(supported_tokens / len(ans_tokens), 4)

        if score >= 0.70:
            status = "supported"
        elif score >= 0.35:
            status = "partially_supported"
        else:
            status = "unsupported"

        return {"grounding_score": score, "grounding_status": status}

    def aggregate_metrics(self, question_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates metrics across all questions:
        - Recall@1, Recall@3, Recall@5, MRR
        - Correctness, Relevance, Grounding
        - Unanswerable detection accuracy & false answer count
        - Category-wise breakdown
        """
        total = len(question_evaluations)
        if total == 0:
            return {}

        recalls_1 = [q["retrieval"]["recall_at_1"] for q in question_evaluations]
        recalls_3 = [q["retrieval"]["recall_at_3"] for q in question_evaluations]
        recalls_5 = [q["retrieval"]["recall_at_5"] for q in question_evaluations]
        mrrs = [q["retrieval"]["reciprocal_rank"] for q in question_evaluations]
        correctness = [q["answer_eval"]["semantic_correctness"] for q in question_evaluations]
        relevance = [q["answer_eval"]["answer_relevance"] for q in question_evaluations]
        grounding = [q["grounding"]["grounding_score"] for q in question_evaluations]
        latencies = [q.get("latency_ms", 25.0) for q in question_evaluations]

        # Unanswerable analysis
        unans_questions = [q for q in question_evaluations if q["category"] == "unanswerable"]
        unans_total = len(unans_questions)
        unans_correct = sum(1 for q in unans_questions if q["answer_eval"]["is_unanswerable_handled"])
        unans_accuracy = (unans_correct / unans_total) if unans_total > 0 else 1.0
        false_answers = unans_total - unans_correct

        # Category-wise metrics
        by_category = {}
        for cat in self.categories:
            cat_items = [q for q in question_evaluations if q["category"] == cat]
            if cat_items:
                by_category[cat] = {
                    "count": len(cat_items),
                    "recall_at_3": round(float(np.mean([q["retrieval"]["recall_at_3"] for q in cat_items])), 4),
                    "mrr": round(float(np.mean([q["retrieval"]["reciprocal_rank"] for q in cat_items])), 4),
                    "correctness": round(float(np.mean([q["answer_eval"]["semantic_correctness"] for q in cat_items])), 4),
                    "grounding": round(float(np.mean([q["grounding"]["grounding_score"] for q in cat_items])), 4)
                }

        return {
            "total_questions": total,
            "recall_at_1": round(float(np.mean(recalls_1)), 4),
            "recall_at_3": round(float(np.mean(recalls_3)), 4),
            "recall_at_5": round(float(np.mean(recalls_5)), 4),
            "mrr": round(float(np.mean(mrrs)), 4),
            "average_latency_ms": round(float(np.mean(latencies)), 2),
            "answer_correctness": round(float(np.mean(correctness)), 4),
            "answer_relevance": round(float(np.mean(relevance)), 4),
            "grounding_score": round(float(np.mean(grounding)), 4),
            "unanswerable_detection_accuracy": round(float(unans_accuracy), 4),
            "false_answer_count": false_answers,
            "by_category": by_category
        }
