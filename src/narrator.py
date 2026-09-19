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
    "You will be given a dataset profile, preprocessing steps taken, and "
    "model comparison results including per-model metrics and a final "
    "justification. Write a clear, plain-English summary for a student "
    "audience. Rules: only describe numbers/decisions already provided, "
    "never invent or estimate anything; do not suggest further steps; "
    "keep it to 4-6 short paragraphs; no markdown headers."
)


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
        lines.append(f"Metric used: {result.metric_used}")
        lines.append("Model comparison:")
        for r in result.candidate_results:
            lines.append(
                f"- {r.model_name}: accuracy={round(r.accuracy,4)}, precision={round(r.precision,4)}, "
                f"recall={round(r.recall,4)}, f1={round(r.f1,4)}, roc_auc={round(r.roc_auc,4) if r.roc_auc==r.roc_auc else 'N/A'}"
            )
        lines.append(f"Baseline score: {round(result.baseline_score, 4)}")
        lines.append(f"Winner: {result.best_model_name}, improvement over baseline: {round(result.improvement_over_baseline, 4)}")
        lines.append(f"Justification: {result.final_justification}")
        return "\n".join(lines)

    def narrate(self, profile: DatasetProfile, plan: PreprocessingPlan, result: SelectionResult) -> str:
        prompt = self.build_prompt(profile, plan, result)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content