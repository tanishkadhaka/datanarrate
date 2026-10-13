"""
DataNarrate — Streamlit UI
Upload a dataset, pick ANY column as the target, and watch the full pipeline
run: target preparation -> profile -> preprocessing (with proof) -> baseline
and model comparison -> narration.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.profiler import DataProfiler
from src.preprocessor import PreprocessingDecider
from src.preprocessing_proof import build_preprocessing_proof
from src.model_selector import ModelSelector, InsufficientDataError
from src.narrator import Narrator
from src.storage import RunStorage
from src.significance import paired_significance_test
from src.target_preparer import prepare_target, InvalidTargetError

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


def _before_after_table(before: dict, after: dict) -> pd.DataFrame:
    keys = list(dict.fromkeys(list(before.keys()) + list(after.keys())))
    return pd.DataFrame({
        "statistic": keys,
        "before": [str(before.get(k, "-")) for k in keys],
        "after": [str(after.get(k, "-")) for k in keys],
    })


def render_preprocessing_proof(proof):
    n_checks = sum(len(p.checks) for p in proof.columns)
    n_passed = sum(1 for p in proof.columns for _, ok, _ in p.checks if ok)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Columns examined", len(proof.columns))
    m2.metric("Features: before → after", f"{proof.n_features_before} → {proof.n_features_after}")
    m3.metric("Missing cells left", proof.nan_cells_after)
    m4.metric("Checks passed", f"{n_passed}/{n_checks}")
    for note in proof.notes:
        st.caption(note)

    warnings = [(p.column, w) for p in proof.columns for w in p.warnings]
    if warnings:
        st.warning(
            "Possible hidden missing values:\n\n"
            + "\n".join(f"- **{col}**: {w}" for col, w in warnings)
        )

    st.dataframe(
        pd.DataFrame([
            {
                "Column": p.column,
                "What was done": ", ".join(p.actions),
                "Checks": f"{sum(1 for _, ok, _ in p.checks if ok)}/{len(p.checks)} passed",
                "Status": "✅" if p.all_passed else "❌",
            }
            for p in proof.columns
        ]),
        hide_index=True, width="stretch",
    )

    st.markdown("#### Column by column: reason, evidence and checks")
    for p in proof.columns:
        icon = "✅" if p.all_passed else "❌"
        with st.expander(f"{icon} {p.column} — {', '.join(p.actions)}"):
            st.markdown("**Why**")
            for reason in p.reasons:
                st.write("•", reason)
            if p.before or p.after:
                st.markdown("**Before vs after**")
                st.dataframe(_before_after_table(p.before, p.after), hide_index=True, width="stretch")
            st.markdown("**Checks run on your data**")
            for name, ok, detail in p.checks:
                st.write(("✅ " if ok else "❌ ") + name + (f" — {detail}" if detail else ""))
            for w in p.warnings:
                st.warning(w)
            if p.examples:
                st.markdown("**Example rows**")
                st.dataframe(pd.DataFrame(p.examples), hide_index=True, width="stretch")


def render_baseline(result):
    info = result.baseline_info
    if info is None:
        st.info("No baseline details are available for this run.")
        return

    st.subheader("What the baseline is")
    st.info(info.summary)
    st.metric(f"Baseline {result.metric_used.upper()} score", round(info.score, 4))

    st.subheader("Baseline vs DataNarrate, stage by stage")
    st.dataframe(pd.DataFrame(info.steps), hide_index=True, width="stretch")

    st.subheader("Why this baseline was chosen")
    for line in info.why_chosen:
        st.write("•", line)

    st.subheader("Where the improvement comes from")
    rows, prev = [], None
    for step in result.ladder:
        rows.append({
            "Step": step.label,
            f"{result.metric_used} score": round(step.score, 4),
            "Change": "-" if prev is None else f"{step.score - prev:+.4f}",
            "What changed": step.what_changed,
        })
        prev = step.score
    ladder_df = pd.DataFrame(rows)
    st.dataframe(ladder_df, hide_index=True, width="stretch")

    score_col = f"{result.metric_used} score"
    fig = px.bar(ladder_df, x="Step", y=score_col, text=score_col, template="plotly_dark")
    lo = min(s.score for s in result.ladder)
    hi = max(s.score for s in result.ladder)
    fig.update_yaxes(range=[max(0.0, lo - 0.05), min(1.0, hi + 0.03)])
    st.plotly_chart(fig, width="stretch", key="ladder_chart")
    st.caption("The vertical axis is zoomed in so small differences are visible.")

    if len(result.ladder) == 4:
        s0, s1, s2, s3 = (s.score for s in result.ladder)
        st.info(
            f"Total change over the baseline: {s3 - s0:+.4f}. "
            f"Class balancing accounts for {s1 - s0:+.4f}, DataNarrate's preprocessing decisions for "
            f"{s2 - s1:+.4f}, and choosing a different model for {s3 - s2:+.4f}."
        )

    st.subheader("Caveats")
    for line in info.caveats:
        st.write("•", line)


st.markdown('<p class="main-header">🧠 DataNarrate</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Upload a dataset. Pick what to predict. See exactly what it decides — and the proof.</p>', unsafe_allow_html=True)
st.divider()

uploaded_file = st.file_uploader("Upload a CSV file (up to 1GB)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)

    df = load_csv(uploaded_file)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><h3>{df.shape[0]:,}</h3>Rows</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3>Columns</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>{round(df.isna().sum().sum() / df.size * 100, 2)}%</h3>Missing data</div>', unsafe_allow_html=True)

    with st.expander("Preview data"):
        st.dataframe(df.head(20), width="stretch")

    target_column = st.selectbox(
        "Select the target column (what you want to predict — usually the last column)",
        df.columns,
        index=len(df.columns) - 1,
    )

    if df.shape[0] > 10000:
        sample_mode = st.checkbox(
            f"Fast mode: sample 10,000 of {df.shape[0]:,} rows for a quicker run "
            f"(uncheck for the full dataset — slower but more accurate)",
            value=True,
        )
    else:
        sample_mode = False

    run = st.button("🚀 Run DataNarrate", type="primary", width="stretch")

    if run:
        run_df = df.sample(n=10000, random_state=42) if sample_mode else df

        try:
            run_df, target_notes = prepare_target(run_df, target_column)
        except InvalidTargetError as e:
            st.error(f"Can't use '{target_column}' as the target: {e}")
            st.stop()

        with st.expander("🎯 How the target column was prepared", expanded=True):
            for note in target_notes:
                st.write("•", note)

        with st.spinner("Profiling dataset..."):
            profile = DataProfiler(run_df, target_column=target_column).profile()
        with st.spinner("Deciding preprocessing steps..."):
            plan = PreprocessingDecider(profile).decide()

        plan.summary = [f"Target preparation: {n}" for n in target_notes] + plan.summary
        plan.step_trace = [f"[target: {target_column}] {n}" for n in target_notes] + plan.step_trace

        try:
            with st.spinner("Verifying every preprocessing step on your data..."):
                proof = build_preprocessing_proof(run_df, profile, plan)
        except Exception as e:
            proof = None
            st.warning(f"Could not build the preprocessing proof ({type(e).__name__}: {e}). The rest of the run is unaffected.")

        try:
            with st.spinner("Comparing the baseline and 4 models — this takes a minute or two..."):
                result = ModelSelector(run_df, profile, plan).select()
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

        try:
            with st.spinner("Generating narration..."):
                narration = Narrator().narrate(profile, plan, result)
        except Exception as e:
            narration = f"Narration unavailable ({type(e).__name__}). All decisions and metrics are unaffected."
            st.warning("The narration step failed (check GROQ_API_KEY in .env). Everything else below is still valid.")

        storage = RunStorage()
        run_id = storage.save_run(
            f"{uploaded_file.name} -> {target_column}" + (" (10k sample)" if sample_mode else ""),
            profile, plan, result, narration,
        )

        tab_overview, tab_prep, tab_base, tab_models, tab_cm, tab_why, tab_narr = st.tabs([
            "📋 Overview", "🔧 Preprocessing", "⚖️ Baseline", "📊 Model Comparison",
            "🧩 Confusion Matrices", "💡 Why This Model", "📝 Narration",
        ])

        with tab_overview:
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
            st.caption(
                "The baseline is a fixed logistic regression on a fixed preprocessing recipe. "
                "See the ⚖️ Baseline tab for exactly what it does, why it was chosen, and where the improvement comes from."
            )

            ranked = sorted(result.candidate_results, key=lambda r: getattr(r, result.metric_used), reverse=True)
            if len(ranked) > 1:
                t, p, sig = paired_significance_test(
                    result.fold_scores[ranked[0].model_name], result.fold_scores[ranked[1].model_name]
                )
                sig_text = "statistically significant" if sig else "not statistically significant — could be noise"
                st.info(f"**Significance check:** {ranked[0].model_name} vs {ranked[1].model_name} → p = {round(p,4)} ({sig_text})")

        with tab_prep:
            if proof is not None:
                st.subheader("Proof that each step did what it claims")
                render_preprocessing_proof(proof)
                st.divider()
            st.subheader("Step-by-step preprocessing trace")
            for step in plan.step_trace:
                st.markdown(f'<div class="step-line">{step}</div>', unsafe_allow_html=True)
            st.subheader("Summary")
            for s in plan.summary:
                st.write("•", s)

        with tab_base:
            render_baseline(result)

        with tab_models:
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
            fig2.update_layout(title="Per-fold score spread (cross-validation)", template="plotly_dark")
            st.plotly_chart(fig2, width="stretch", key="fold_spread_boxplot")

        with tab_cm:
            st.caption("Axes show class numbers. See 'Class order' under 'How the target column was prepared' for what each number means.")
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

        with tab_why:
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

        with tab_narr:
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