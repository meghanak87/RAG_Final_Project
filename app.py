"""
Streamlit Application for RAG Document Intelligence & Benchmark Studio.
Pages:
1. Ask Questions
2. Model Comparison
3. Analytics
4. Benchmark Evaluation
"""

import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="RAG Document Intelligence & Benchmark Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

from rag_engine import RAGEngine
from embedding_pipeline import MODEL_CONFIGS
from evaluation_engine import EvaluationEngine
from evaluation.benchmark_runner import run_benchmark, BENCHMARK_FILE, RESULTS_FILE, SOURCE_DOC_FILE
from evaluation.evaluate_benchmark import evaluate_all, METRICS_FILE, REPORT_FILE

# Initialize session state for RAG Engine and cache
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
    # Auto-ingest default source document if available
    if os.path.exists(SOURCE_DOC_FILE):
        with open(SOURCE_DOC_FILE, "rb") as f:
            st.session_state.rag_engine.ingest_document(
                f.read(),
                os.path.basename(SOURCE_DOC_FILE)
            )

# Sidebar Navigation
st.sidebar.title("📄 RAG Studio")
st.sidebar.markdown("**Document Intelligence & Benchmarking**")

page = st.sidebar.radio(
    "Navigation",
    ["Ask Questions", "Model Comparison", "Analytics", "Benchmark Evaluation"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("5 Embedding Models Active:")
st.sidebar.markdown("- 🔷 **BGE-M3** (1024d)\n- 🌐 **Multilingual-E5-large** (1024d)\n- 🇬🇧 **BGE-large-en-v1.5** (1024d)\n- ⚡ **MiniLM** (384d)\n- 📜 **Nomic Embed Text v1.5** (768d)")

# ==========================================
# PAGE 1: ASK QUESTIONS
# ==========================================
if page == "Ask Questions":
    st.title("💬 Ask Questions")
    st.markdown("Query enterprise documents using 5-model parallel vector retrieval with dynamic model selection and grounded Gemini synthesis.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_file is not None:
            if st.button("Process & Index Document"):
                with st.spinner("Processing PDF and building 5 FAISS vector indexes..."):
                    bytes_data = uploaded_file.read()
                    res = st.session_state.rag_engine.ingest_document(bytes_data, uploaded_file.name)
                    st.success(f"Indexed {res['total_pages']} pages and {res['total_chunks']} chunks across all 5 models!")

    with col2:
        doc_name = st.session_state.rag_engine.document_name or "Enterprise_AI_Architecture_and_Governance_Specification_2025.pdf"
        st.info(f"**Current Document:** `{doc_name}`\n\n**Indexed Chunks:** {len(st.session_state.rag_engine.chunks)}\n\n**Status:** {'Ready' if st.session_state.rag_engine.is_indexed else 'Awaiting ingestion'}")

    st.markdown("---")
    query_input = st.text_input("Enter your question:", placeholder="e.g., What are the four linear stages of the core ingestion pipeline?")
    top_k = st.slider("Retrieval Top-K", min_value=1, max_value=10, value=5)

    if st.button("Submit Query", type="primary"):
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running retrieval across 5 models & generating grounded response..."):
                start_t = time.time()
                ans_data = st.session_state.rag_engine.query(query_input, top_k=top_k)
                elapsed = (time.time() - start_t) * 1000.0

                st.markdown("### Answer")
                st.markdown(f"> {ans_data['generated_answer']}")

                # Badges row
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Selected Model", ans_data["selected_model"])
                with b2:
                    st.metric("Selection Score", f"{ans_data['selection_score']:.3f}")
                with b3:
                    st.metric("Grounding Score", f"{ans_data['grounding_score']:.1%}")
                with b4:
                    st.metric("Total Latency", f"{elapsed:.1f} ms")

                st.info(f"**Selection Rationale:** {ans_data['selection_reason']}")

                # Grounded Evidence
                st.markdown("#### Selected Evidence Chunks")
                for c in ans_data["selected_evidence"]:
                    with st.expander(f"Chunk {c.get('chunk_id')} — Page {c.get('page')} (Similarity: {c.get('similarity', 0):.4f})"):
                        st.write(c.get("text"))

# ==========================================
# PAGE 2: MODEL COMPARISON
# ==========================================
elif page == "Model Comparison":
    st.title("⚖️ Model Comparison")
    st.markdown("Inspect side-by-side retrieval performance across all five embedding models for any query.")

    comp_query = st.text_input("Test Query for Model Comparison:", value="Compare the context window token limits of BGE-M3 versus Multilingual-E5-large.")
    if st.button("Run Multi-Model Comparison"):
        with st.spinner("Executing retrieval across all 5 models..."):
            comp_res = st.session_state.rag_engine.comparison_engine.compare_retrieval(comp_query, top_k=5)
            dyn = comp_res["dynamic_selection"]

            st.success(f"Optimal Model Selected: **{dyn['selected_model']}** (Score: {dyn['selection_score']:.4f})")
            st.caption(f"Reason: {dyn['selection_reason']}")

            # Comparison Metrics Table
            evals = dyn["model_evaluations"]
            table_data = []
            for m, dat in evals.items():
                table_data.append({
                    "Embedding Model": m,
                    "Raw Top-1 Sim": dat["raw_top_1"],
                    "Calibrated Z-Score": dat["z_score"],
                    "Normalized Rel": dat["norm_relevance"],
                    "Confidence Margin (s1-s3)": dat["margin_1_3"],
                    "Latency (ms)": MODEL_CONFIGS[m]["latency_ms"],
                    "Composite Score": dat["composite_score"]
                })
            df_comp = pd.DataFrame(table_data)
            st.dataframe(df_comp, use_container_width=True)

            # Retrieved Chunks Tabs
            st.markdown("#### Retrieved Chunks by Model")
            tabs = st.tabs(list(comp_res["results_by_model"].keys()))
            for tab, (m_name, chunks) in zip(tabs, comp_res["results_by_model"].items()):
                with tab:
                    for c in chunks:
                        st.markdown(f"**Rank {c['rank']} (Score: {c['similarity']:.4f})** — *Page {c['page']}, {c['chunk_id']}*")
                        st.caption(c["text"])
                        st.markdown("---")

# ==========================================
# PAGE 3: ANALYTICS
# ==========================================
elif page == "Analytics":
    st.title("📊 RAG Analytics & Distribution Calibration")
    st.markdown("Deep dive into score manifolds, calibration z-scores, and latency budgets.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Embedding Manifold Distributions")
        st.markdown("""
        **The E5 Score Distribution Anomaly:**
        Multilingual-E5-large models exhibit an elevated baseline cosine distribution (mean 0.84, min ~0.78),
        whereas BGE-large-en operates at mean 0.70 and MiniLM at 0.61.
        
        Comparing raw cosine similarities creates an artificial **100% win rate for E5**.
        Our calibrated selection algorithm solves this via:
        1. **Z-Score Normalization**
        2. **Discriminative Confidence Margin** ($s_1 - s_3$)
        3. **Query Topology Alignment**
        """)
        
        # Plot typical score distributions
        m_names = list(MODEL_CONFIGS.keys())
        means = [MODEL_CONFIGS[m]["mean_baseline"] for m in m_names]
        stds = [MODEL_CONFIGS[m]["std_baseline"] for m in m_names]
        df_dist = pd.DataFrame({"Model": m_names, "Baseline Mean": means, "Std Dev": stds})
        st.bar_chart(df_dist.set_index("Model"))

    with col2:
        st.subheader("Inference Latency Profile (ms)")
        latencies = [MODEL_CONFIGS[m]["latency_ms"] for m in m_names]
        df_lat = pd.DataFrame({"Model": m_names, "Latency (ms)": latencies})
        st.bar_chart(df_lat.set_index("Model"))

# ==========================================
# PAGE 4: BENCHMARK EVALUATION
# ==========================================
elif page == "Benchmark Evaluation":
    st.title("🎯 Benchmark Evaluation Dashboard")
    st.markdown("Official 50-question Ground Truth Evaluation across all 5 Embedding Models and Dynamic Selection.")

    # Load existing benchmark or run
    report_data = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            report_data = json.load(f)

    c_btn1, c_btn2, c_spacer = st.columns([1.5, 1.5, 4])
    with c_btn1:
        if st.button("🚀 Run Full 50-Q Benchmark", type="primary"):
            with st.spinner("Executing 50 benchmark queries through RAG pipeline..."):
                run_benchmark()
                report_data = evaluate_all()
                st.success("Benchmark completed and evaluated successfully!")
                st.rerun()

    with c_btn2:
        if st.button("🔄 Re-compute Metrics Only"):
            report_data = evaluate_all()
            st.success("Metrics re-calculated!")
            st.rerun()

    if report_data:
        dyn_summary = report_data["dynamic_selection"]
        models_data = report_data["models"]

        # 1. KPI Headline Metrics (7 requested metrics)
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        results_count = len(results) if os.path.exists(RESULTS_FILE) and (results := json.load(open(RESULTS_FILE, "r", encoding="utf-8"))) else dyn_summary.get("total_questions", 50)
        with k1:
            st.metric("Total Questions", dyn_summary["total_questions"])
        with k2:
            st.metric("Completed Questions", results_count)
        with k3:
            st.metric("Recall@3", f"{dyn_summary['recall_at_3']:.1%}")
        with k4:
            st.metric("MRR", f"{dyn_summary['mrr']:.3f}")
        with k5:
            st.metric("Answer Correctness", f"{dyn_summary['answer_correctness']:.1%}")
        with k6:
            st.metric("Grounding Score", f"{dyn_summary['grounding_score']:.1%}")
        with k7:
            st.metric("Unanswerable Acc.", f"{dyn_summary['unanswerable_detection_accuracy']:.1%}")

        st.markdown("---")

        # 2. Comparison Table & Charts: All 5 Models + Dynamic Selection
        st.subheader("Model Benchmark Comparison: All 5 Models vs. Dynamic Selection")
        comp_rows = []
        all_eval_models = list(models_data.keys()) + ["Dynamic Selection"]
        chart_data = []

        for m in all_eval_models:
            m_stat = dyn_summary if m == "Dynamic Selection" else models_data[m]
            comp_rows.append({
                "Model / Strategy": m,
                "Recall@1": f"{m_stat['recall_at_1']:.1%}",
                "Recall@3": f"{m_stat['recall_at_3']:.1%}",
                "Recall@5": f"{m_stat['recall_at_5']:.1%}",
                "MRR": f"{m_stat['mrr']:.3f}",
                "Latency (ms)": f"{m_stat['average_latency_ms']:.1f}",
                "Correctness": f"{m_stat['answer_correctness']:.1%}",
                "Grounding": f"{m_stat['grounding_score']:.1%}",
                "Unanswerable Acc": f"{m_stat['unanswerable_detection_accuracy']:.1%}"
            })
            chart_data.append({
                "Model": m,
                "Recall@3": m_stat["recall_at_3"] * 100,
                "MRR": m_stat["mrr"] * 100,
                "Latency_ms": m_stat["average_latency_ms"]
            })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

        # Comparison Charts
        c_col1, c_col2 = st.columns(2)
        df_chart = pd.DataFrame(chart_data).set_index("Model")
        with c_col1:
            st.caption("Recall@3 (%) & MRR (x100) by Model")
            st.bar_chart(df_chart[["Recall@3", "MRR"]])
        with c_col2:
            st.caption("Average Retrieval Latency (ms) by Model")
            st.bar_chart(df_chart[["Latency_ms"]])

        # 3. Dynamic Model Selection Win Rate Breakdown (Demonstrating fixed E5 bias)
        st.subheader("Dynamic Model Selection Win Distribution")
        dist = dyn_summary.get("selection_distribution", {})
        if dist:
            dist_df = pd.DataFrame([
                {"Model": m, "Selected Queries": dat["count"], "Percentage": f"{dat['percentage']}%"}
                for m, dat in dist.items()
            ])
            st.dataframe(dist_df, use_container_width=True)

        # 4. Category-wise Breakdown
        st.subheader("Performance Breakdown by Category")
        cat_data = report_data.get("category_breakdown", {})
        if cat_data:
            cat_df = pd.DataFrame([
                {
                    "Category": cat.capitalize(),
                    "Count": dat["count"],
                    "Recall@3": f"{dat['recall_at_3']:.1%}",
                    "MRR": f"{dat['mrr']:.3f}",
                    "Correctness": f"{dat['correctness']:.1%}",
                    "Grounding": f"{dat['grounding']:.1%}"
                }
                for cat, dat in cat_data.items()
            ])
            st.dataframe(cat_df, use_container_width=True)

        # 5. Individual Question Inspector (Step 13)
        st.markdown("---")
        st.subheader("🔍 Individual Question Inspector (50 Questions)")
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            raw_results = json.load(f)

        q_ids = [f"{item['id']}: {item['question'][:60]}..." for item in raw_results]
        selected_idx = st.selectbox("Select Benchmark Question to Inspect:", range(len(q_ids)), format_func=lambda i: q_ids[i])

        item = raw_results[selected_idx]

        # Calculate specific metrics for this individual question
        eval_inst = EvaluationEngine()
        q_ret_metrics = eval_inst.evaluate_retrieval_metrics(
            retrieved_chunks=item.get("selected_evidence", []), 
            expected_evidence=item.get("expected_evidence", ""), 
            page=item.get("page"), 
            category=item.get("category", ""),
            source=item.get("source", ""),
            retrieval_latency=item.get("total_latency_ms", 0.0)
        )
        q_ans_metrics = eval_inst.evaluate_answer_correctness(
            item.get("generated_answer", ""), 
            item.get("ground_truth", ""), 
            item.get("category", "")
        )

        # Question-level Metrics Cards
        qm1, qm2, qm3, qm4, qm5, qm6 = st.columns(6)
        with qm1:
            st.metric("Recall@1", "1.0 (Hit)" if q_ret_metrics["recall_at_1"] == 1.0 else "0.0 (Miss)")
        with qm2:
            st.metric("Recall@3", "1.0 (Hit)" if q_ret_metrics["recall_at_3"] == 1.0 else "0.0 (Miss)")
        with qm3:
            st.metric("Recall@5", "1.0 (Hit)" if q_ret_metrics["recall_at_5"] == 1.0 else "0.0 (Miss)")
        with qm4:
            st.metric("Reciprocal Rank (MRR)", f"{q_ret_metrics['reciprocal_rank']:.3f}")
        with qm5:
            st.metric("Answer Correctness", f"{q_ans_metrics['semantic_correctness']:.1%}")
        with qm6:
            st.metric("Grounding Status", f"{item.get('grounding_status', 'N/A')}", f"{item.get('grounding_score', 0):.0%}")

        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Category:** `{item['category']}` | **Expected Page:** `{item['page']}` | **Source:** `{item['source']}`")
            st.markdown(f"**Ground Truth:**\n> {item['ground_truth']}")
            st.markdown(f"**Expected Evidence:**\n`{item['expected_evidence'] or 'None (Unanswerable)'}`")

        with col_q2:
            st.markdown(f"**Selected Dynamic Model:** 🏆 `{item['selected_model']}`")
            st.markdown(f"**Selection Score:** `{item['selection_score']:.4f}`")
            st.caption(f"**Selection Rationale:** {item['selection_reason']}")
            st.markdown(f"**Generated Answer:**\n> {item['generated_answer']}")
            st.markdown(f"**Latency:** `{item['total_latency_ms']:.1f} ms`")

        # Side-by-side evidence inspection for all 5 models
        st.markdown("##### Retrieved Evidence Across All 5 Models")
        tabs = st.tabs(["BGE-M3", "Multilingual-E5-large", "BGE-large-en-v1.5", "MiniLM", "Nomic Embed Text v1.5", "Selected Evidence"])
        models_list = ["BGE-M3", "Multilingual-E5-large", "BGE-large-en-v1.5", "MiniLM", "Nomic Embed Text v1.5"]
        
        for tab, m_name in zip(tabs[:5], models_list):
            with tab:
                m_retrieved = item["models_retrieval"][m_name]["top_3"]
                for r_chunk in m_retrieved:
                    st.markdown(f"**Rank {r_chunk['rank']}** — Page {r_chunk['page']} | Score: {r_chunk['similarity']:.4f}")
                    st.caption(r_chunk["text"])
                    st.markdown("---")
        
        with tabs[5]:
            for s_chunk in item["selected_evidence"]:
                st.markdown(f"**Page {s_chunk.get('page')}** | Chunk {s_chunk.get('chunk_id')} | Score: {s_chunk.get('similarity', 0):.4f}")
                st.caption(s_chunk.get("text"))
                st.markdown("---")

    else:
        st.info("Benchmark has not been run yet. Click 'Run Full 50-Q Benchmark' above to execute all 50 questions through the real RAG pipeline.")
