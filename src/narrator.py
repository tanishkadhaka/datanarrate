"""
Module 4 — Narration Layer
Describes the outputs of Modules 1-3 in plain English. Strictly grounded —
only describes numbers/decisions that already exist, never invents anything.
"""
import os
from groq import Groq
from dotenv import load_dotenv

from src.profiler import DatasetProfile
from src.preprocessor import PreprocessingPlan
from src.model_selector import SelectionResult

load_dotenv()
DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are a narration assistant for an automated machine learning tool. "
    "You will be given a dataset profile, preprocessing steps taken, model comparison "
    "results, a baseline definition and an improvement breakdown. Write a clear, "
    "plain-English summary for a student audience. Rules: only describe numbers and "
    "decisions already provided, never invent or estimate anything; describe the "
    "baseline exactly as defined in the Baseline definition and never as a "
    "majority-class or random guesser; explain where the improvement comes from using "
    "the breakdown; do not suggest further steps; keep it to 4-6 short paragraphs; "
    "no markdown headers. Mention how outliers were handled (which columns were "
    "clipped, and how many percent of their values were outliers) if outlier "
    "information is given."
)

MAX_OUTLIER_LINES = 10   # keep the prompt short on datasets with many numeric columns


class Narrator:
    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        self.client = Groq(api_key=key)
        self.model = model

    def build_prompt(self, profile: DatasetProfile, plan: PreprocessingPlan, result: SelectionResult) -> str:
        lines = [
            f"Dataset: {profile.n_rows} rows, {profile.n_cols} columns, "
            f"{profile.n_classes} classes, imbalanced={profile.is_imbalanced}.",
            "",
            "Preprocessing steps taken:",
        ]
        lines.extend(f"- {s}" for s in plan.summary)
        lines.append("")
        lines.extend(self._outlier_lines(profile, plan))
        lines.append("")
        lines.append(f"Metric used: {result.metric_used}")
        lines.append("Model comparison:")
        for r in result.candidate_results:
            lines.append(
                f"- {r.model_name}: accuracy={round(r.accuracy,4)}, precision={round(r.precision,4)}, "
                f"recall={round(r.recall,4)}, f1={round(r.f1,4)}, roc_auc={round(r.roc_auc,4) if r.roc_auc==r.roc_auc else 'N/A'}"
            )
        lines.append(f"Baseline score: {round(result.baseline_score, 4)}")
        if result.baseline_info is not None:
            lines.append("Baseline definition: " + result.baseline_info.summary)
        if result.ladder:
            lines.append("Improvement breakdown (score after each added change):")
            for step in result.ladder:
                lines.append(f"- {step.label}: {round(step.score, 4)} ({step.what_changed})")
        lines.append(f"Winner: {result.best_model_name}, improvement over baseline: {round(result.improvement_over_baseline, 4)}")
        lines.append(f"Justification: {result.final_justification}")
        return "\n".join(lines)

    def _outlier_lines(self, profile: DatasetProfile, plan: PreprocessingPlan) -> list:
        """Facts about rule R8 (outlier clipping), one line per clipped column."""
        pct_of = {c.name: c.outlier_pct for c in profile.columns}
        clipped = [d for d in plan.column_decisions if d.outlier_action == "clip"]
        n_numeric = sum(1 for d in plan.column_decisions if d.scaling_action == "standard_scale")

        lines = ["Outlier handling (rule R8, IQR method; rows are never removed, extreme values are clipped):"]
        for d in clipped[:MAX_OUTLIER_LINES]:
            how = "1st/99th percentiles because the IQR is 0" if d.outlier_bounds == "percentile" else "IQR fences"
            lines.append(
                f"- {d.column}: {pct_of.get(d.column)}% of values were outliers, clipped to "
                f"[{round(d.clip_lower, 2)}, {round(d.clip_upper, 2)}] using {how}"
            )
        if len(clipped) > MAX_OUTLIER_LINES:
            lines.append(f"- ...and {len(clipped) - MAX_OUTLIER_LINES} more clipped columns")
        lines.append(
            f"- {len(clipped)} of {n_numeric} numeric columns were clipped; "
            f"the rest had too few outliers (under 1%) or no usable range."
        )
        return lines

    def narrate(self, profile: DatasetProfile, plan: PreprocessingPlan, result: SelectionResult) -> str:
        prompt = self.build_prompt(profile, plan, result)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content