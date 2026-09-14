"""
Module 4 — Narration Layer
Takes the outputs of Modules 1–3 (profile, preprocessing plan, model selection
result) and produces a plain-English explanation via an LLM. This module is
strictly a describer, not a decider — it is never allowed to introduce any
number, model name, or claim that isn't already present in the inputs.
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
    "You will be given a dataset profile, a set of preprocessing decisions, "
    "and model selection results. Write a clear, plain-English summary for a "
    "student audience explaining what was done and why. "
    "Rules you must follow strictly: "
    "1. Only describe numbers, decisions, and model names that appear in the "
    "provided data — never invent or estimate anything. "
    "2. Do not suggest any additional preprocessing steps or models beyond "
    "what was already decided. "
    "3. Keep it to 3-5 short paragraphs. "
    "4. Do not use markdown headers."
)


class Narrator:
    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        self.client = Groq(api_key=key)
        self.model = model

    def build_prompt(
        self,
        profile: DatasetProfile,
        plan: PreprocessingPlan,
        result: SelectionResult,
    ) -> str:
        lines = [
            f"Dataset: {profile.n_rows} rows, {profile.n_cols} columns.",
            f"Task type: {profile.task_type}, {profile.n_classes} classes.",
            f"Imbalanced: {profile.is_imbalanced}.",
            "",
            "Preprocessing decisions summary:",
        ]
        lines.extend(f"- {s}" for s in plan.summary)
        lines.append("")
        lines.append(f"Evaluation metric used: {result.metric_used}")
        lines.append("Candidate models and their cross-validated scores:")
        for c in result.candidate_results:
            lines.append(f"- {c.model_name}: {round(c.mean_score, 4)} (std {round(c.std_score, 4)})")
        lines.append(f"Best model selected: {result.best_model_name} (score {round(result.best_score, 4)})")
        lines.append(f"Fixed baseline pipeline score: {round(result.baseline_score, 4)}")
        lines.append(f"Improvement over baseline: {round(result.improvement_over_baseline, 4)}")
        return "\n".join(lines)

    def narrate(
        self,
        profile: DatasetProfile,
        plan: PreprocessingPlan,
        result: SelectionResult,
    ) -> str:
        prompt = self.build_prompt(profile, plan, result)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content