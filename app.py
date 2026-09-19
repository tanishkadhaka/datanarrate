"""
DataNarrate — Streamlit UI
Upload a dataset, pick the target column, and watch the full pipeline run:
profile -> preprocessing decisions -> model comparison -> narration.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.model_selector import ModelSelector, InsufficientDataError
from src.narrator import Narrator
from src.storage import RunStorage
from src.significance import paired_significance_test

st.set_page_config(page_title="DataNarrate", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #6C5CE7, #00D2D3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtext { color: #9CA3AF; font-size: 1.05rem; margin-top: 0; }
.metric-card {
    background: #1A1D27; border-radius: 12px; padding: 1.2rem;
    border: 1px solid #2A2E3A; text-align: center;
}
.step-line {
    background: #161A23; border-left: 3px solid #6C5CE7;
    padding: 0.5rem 0.9rem; margin-bottom: 0.4rem; border-radius: 6px;
    font-size: 0.92rem;
}
.winner-banner {
    background: linear-gradient(90deg, #6C5CE7 0%, #00D2D3 100%);
    padding: 1.2rem 1.5rem; border-radius: 12px; color: white;
    font-size: 1.1rem; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧠 DataNarrate</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Upload a classification dataset. See exactly what it decides — and why.</p>', unsafe_allow_html=True)
st.divider()

uploaded_file = st.file_uploader("Upload a CSV file (up to 1GB)", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if uploaded_file is not None:
    df = load_csv(uploaded_file)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><h3>{df.shape[0]:,}</h3>Rows</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3>Columns</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>{round(df.isna().sum().sum() / df.size * 100, 2)}%</h3>Missing data</div>', unsafe_allow_html=True)

    with st.expander("Preview data"):
        st.dataframe(df.head(20), width="stretch")

    target_column = st.selectbox("Select the target (label) column", df.columns)
    run = st.button("🚀 Run DataNarrate", type="primary", width="stretch")

    if run:
        with st.spinner("Profiling dataset..."):
            profile = DataProfiler(df, target_column=target_column).profile()
        with st.spinner("Deciding preprocessing steps..."):
            plan = PreprocessingDecider(profile).decide()

        try:
            with st.spinner("Comparing 4 models — this takes a minute or two..."):
                result = ModelSelector(df, profile, plan).select()
        except InsufficientDataError as e:
            st.error(f"Can't run model comparison: {e}")
            st.stop()

        if result.cv_folds_used < 5:
            st.warning(
                f"Used {result.cv_folds_used}-fold cross-validation instead of the usual 5, "
                f"because the smallest class in your target column only has "
                f"{result.cv_folds_used} examples. Results are still valid, but less stable "
                f"than they'd be with more data."
            )

        with st.spinner("Generating narration..."):
            narration = Narrator().narrate(profile, plan, result)

        storage = RunStorage()
        run_id = storage.save_run(uploaded_file.name, profile, plan, result, narration)

        tabs = st.tabs(["📋 Overview", "🔧 Preprocessing", "📊 Model Comparison", "🧩 Confusion Matrices", "💡 Why This Model", "📝 Narration"])

        with tabs[0]:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Classes", profile.n_classes)
            k2.metric("Imbalanced", "Yes" if profile.is_imbalanced else "No")
            k3.metric("Metric used", result.metric_used.upper())
            k4.metric("Columns dropped", len(result.dropped_columns))

            st.markdown(
                f'<div class="winner-banner">🏆 <b>{result.best_model_name}</b> selected — '
                f'{result.metric_used}: <b>{round(result.best_score,4)}</b> vs baseline '
                f'<b>{round(result.baseline_score,4)}</b> '
                f'({"+" if result.improvement_over_baseline >= 0 else ""}{round(result.improvement_over_baseline,4)})</div>',
                unsafe_allow_html=True,
            )

            ranked = sorted(result.candidate_results, key=lambda r: getattr(r, result.metric_used), reverse=True)
            if len(ranked) > 1:
                t, p, sig = paired_significance_test(
                    result.fold_scores[ranked[0].model_name], result.fold_scores[ranked[1].model_name]
                )
                sig_text = "statistically significant" if sig else "not statistically significant — could be noise"
                st.info(f"**Significance check:** {ranked[0].model_name} vs {ranked[1].model_name} → p = {round(p,4)} ({sig_text})")

        with tabs[1]:
            st.subheader("Step-by-step preprocessing trace")
            for step in plan.step_trace:
                st.markdown(f'<div class="step-line">{step}</div>', unsafe_allow_html=True)
            st.subheader("Summary")
            for s in plan.summary:
                st.write("•", s)

        with tabs[2]:
            metrics_df = pd.DataFrame([
                {"Model": r.model_name, "Accuracy": r.accuracy, "Precision": r.precision,
                 "Recall": r.recall, "F1": r.f1, "ROC-AUC": r.roc_auc}
                for r in result.candidate_results
            ])
            st.dataframe(metrics_df.round(4), width="stretch")

            melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig = px.bar(
                melted, x="Model", y="Score", color="Metric", barmode="group",
                title="Model comparison across all metrics", template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            st.plotly_chart(fig, width="stretch", key="metrics_bar_chart")

            fig2 = go.Figure()
            for name, scores in result.fold_scores.items():
                fig2.add_trace(go.Box(y=scores, name=name))
            fig2.update_layout(title="Per-fold score spread (5-fold CV)", template="plotly_dark")
            st.plotly_chart(fig2, width="stretch", key="fold_spread_boxplot")

        with tabs[3]:
            cols = st.columns(2)
            for i, r in enumerate(result.candidate_results):
                with cols[i % 2]:
                    st.write(f"**{r.model_name}**")
                    cm_fig = px.imshow(
                        r.confusion_matrix, text_auto=True, color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        template="plotly_dark",
                    )
                    cm_fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(cm_fig, width="stretch", key=f"cm_{r.model_name}")

        with tabs[4]:
            st.subheader("Final justification")
            st.success(result.final_justification)

            st.subheader("Per-model gains and limitations")
            for r in result.candidate_results:
                with st.expander(f"{r.model_name} — {result.metric_used}: {round(getattr(r, result.metric_used), 4)}"):
                    st.write("**Gains:**")
                    for g in r.gains:
                        st.write("✅", g)
                    st.write("**Limitations:**")
                    for l in r.limitations:
                        st.write("⚠️", l)

        with tabs[5]:
            st.write(narration)

        st.success(f"Run saved to database (#{run_id})")

st.divider()
st.subheader("📂 Past Runs")
storage = RunStorage()
runs = storage.list_runs()
if runs:
    history_df = pd.DataFrame([
        {"Dataset": r.dataset_name, "Best Model": r.best_model_name,
         "Score": round(r.best_score, 4), "Baseline": round(r.baseline_score, 4),
         "Improvement": round(r.improvement_over_baseline, 4), "Time": r.timestamp}
        for r in runs
    ])
    st.dataframe(history_df, width="stretch")
else:
    st.write("No runs saved yet.")