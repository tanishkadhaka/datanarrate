"""
Narration fact-checker
Checks the plain-English narration (written by an LLM) against the numbers and
decisions the pipeline actually computed, and flags anything that does not match.

What it checks
  1. Numbers: every figure in the text must match a value from the run
     (scores, percentages, counts, ladder steps, differences between models),
     allowing for rounding and for percent vs fraction (0.82 vs 82%).
  2. Metric claims: "<model> ... <metric> ... <number>" must match that model's
     score for that metric.
  3. Comparisons: "A outperformed B" / "A was lower than B" must agree with the scores.
  4. Winner: sentences naming the best/selected model must name the real winner.
  5. Baseline: the baseline is a fixed logistic regression, never a
     "majority-class" or "random" guesser.
  6. Columns: a column described as dropped must really have been dropped.

Limits (be upfront about these): it does not understand arbitrary prose. A wrong
number that happens to equal another value from the run can slip through, and
claims it has no rule for (causes, advice) are not checked.

Usage:
    report = check_narration(text, facts)   # facts = the API's result payload (a dict)
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

MODEL_ALIASES = {
    "logistic_regression": ["logistic regression", "logistic_regression"],
    "random_forest": ["random forest", "random_forest"],
    "gradient_boosting": ["gradient boosting", "gradient_boosting"],
    "svm": ["svm", "support vector"],
    "knn": ["knn", "k-nearest", "nearest neighbo"],
    "naive_bayes": ["naive bayes", "naive_bayes", "naïve bayes"],
    "mlp": ["mlp", "neural network"],
    "voting_ensemble": ["voting ensemble", "voting_ensemble", "voting classifier"],
}
METRIC_WORDS = [("roc-auc", "roc_auc"), ("roc_auc", "roc_auc"), ("auc", "roc_auc"),
                ("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall"),
                ("f1", "f1"), ("f-1", "f1")]
POSITIVE = ["outperform", "beat", "better than", "higher than", "ahead of", "exceed", "surpass", "top of"]
NEGATIVE = ["worse than", "lower than", "behind", "trailed", "below", "underperform", "fell short of"]
PASSIVE_POSITIVE = ["outperformed by", "beaten by", "surpassed by", "exceeded by"]   # "A was outperformed by B"
WINNER_WORDS = ["winner", "best", "selected", "chosen", "top model", "top-performing", "champion"]

NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(%?)(?![A-Za-z0-9])")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+")
# LLMs often group thousands with a (thin / no-break) space: "416 810", "1 980.21"
SPACE_GROUP_RE = re.compile(r"(?<=\d)[    ](?=\d{3}(?!\d))")
STRING_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# ---------- helpers ----------

def _leaves(obj: Any) -> Iterable[float]:
    """Every number found anywhere inside a nested dict/list."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if obj == obj and abs(obj) != float("inf"):
            yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)


def _string_numbers(obj: Any) -> Iterable[float]:
    """Numbers written inside text values, e.g. the class labels '<=50K' and '>50K'."""
    if isinstance(obj, str):
        for m in STRING_NUMBER_RE.finditer(obj):
            yield float(m.group())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _string_numbers(k)
            yield from _string_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _string_numbers(v)


def _derived_values(selection: Dict[str, Any]) -> List[float]:
    """Differences the narration is allowed to mention (model vs model, ladder step vs step)."""
    out: List[float] = []
    models = selection.get("candidate_results", [])
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        vals = [m.get(key) for m in models if isinstance(m.get(key), (int, float))]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                out.append(abs(vals[i] - vals[j]))
    ladder = [s.get("score") for s in selection.get("ladder", []) if isinstance(s.get("score"), (int, float))]
    for i in range(len(ladder)):
        for j in range(i + 1, len(ladder)):
            out.append(abs(ladder[i] - ladder[j]))
    best, base = selection.get("best_score"), selection.get("baseline_score")
    if isinstance(best, (int, float)) and isinstance(base, (int, float)):
        out.append(abs(best - base))
    return out


def _matches(value: float, decimals: int, is_pct: bool, allowed: List[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for a in allowed:
        a = abs(a)
        if is_pct:
            if abs(a * 100 - value) <= tol or abs(a - value) <= tol:
                return True
        else:
            if abs(a - value) <= tol or abs(a * 100 - value) <= tol:
                return True
    return False


def _models_in(sentence: str) -> List[Tuple[int, str]]:
    low = sentence.lower()
    found: List[Tuple[int, str]] = []
    for key, aliases in MODEL_ALIASES.items():
        for alias in aliases:
            for m in re.finditer(re.escape(alias), low):
                found.append((m.start(), key))
    found.sort()
    dedup: List[Tuple[int, str]] = []
    for pos, key in found:
        if not dedup or dedup[-1][1] != key or pos - dedup[-1][0] > 3:
            dedup.append((pos, key))
    return dedup


def _metric_in(sentence: str) -> Optional[str]:
    low = sentence.lower()
    hits = {key for word, key in METRIC_WORDS if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", low)}
    return hits.pop() if len(hits) == 1 else None


def _finding(kind: str, severity: str, sentence: str, detail: str) -> Dict[str, str]:
    return {"type": kind, "severity": severity, "sentence": sentence.strip(), "detail": detail}


# ---------- main entry point ----------

def check_narration(text: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    selection = facts.get("selection", {})
    metric_used = selection.get("metric_used", "accuracy")
    models = {m["model_name"]: m for m in selection.get("candidate_results", [])}
    best = selection.get("best_model_name")

    sources = {k: facts.get(k) for k in
               ("selection", "missing_report", "missing_handling", "profile", "preprocessing")}
    allowed = list(_leaves(sources))
    allowed += list(_string_numbers(sources))
    allowed += _derived_values(selection)

    text = SPACE_GROUP_RE.sub("", text or "")
    sentences = [s for s in SENT_SPLIT.split(text) if s and s.strip()]
    findings: List[Dict[str, str]] = []
    checked = verified = 0

    for sent in sentences:
        low = sent.lower()

        # 1. numbers
        for m in NUMBER_RE.finditer(sent):
            raw, pct = m.group(1), m.group(2) == "%"
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            if decimals == 0 and not pct and value <= 10:
                continue                      # small counts ("8 models", "5 folds") are not worth checking
            if decimals == 0 and not pct and 1900 <= value <= 2100:
                continue                      # years
            checked += 1
            if _matches(value, decimals, pct, allowed):
                verified += 1
            else:
                findings.append(_finding(
                    "number", "high", sent,
                    f"{raw}{'%' if pct else ''} does not match any number computed in this run"))

        # 2. "<model> ... <metric> ... <number>" must match that model's score
        found = _models_in(sent)
        keys = {k for _, k in found}
        metric = _metric_in(sent)
        if len(keys) == 1 and metric and "baseline" not in low and "improv" not in low:
            model = models.get(next(iter(keys)))
            score = model.get(metric) if model else None
            nums = [(float(n.group(1).replace(",", "")), n.group(2) == "%", n.group(1)) for n in NUMBER_RE.finditer(sent)]
            scorelike = [(v, p, r) for v, p, r in nums if (p and 0 < v <= 100) or (not p and 0 < v <= 1 and "." in r)]
            if isinstance(score, (int, float)) and scorelike:
                ok = any(_matches(v, len(r.split(".")[1]) if "." in r else 0, p, [score]) for v, p, r in scorelike)
                if not ok:
                    findings.append(_finding(
                        "metric", "high", sent,
                        f"{next(iter(keys))}'s {metric} was {score:.4f}, which differs from the figure(s) in this sentence"))

        # 3. comparisons between two models
        if len(found) >= 2:
            a, b = found[0][1], found[1][1]
            cmp_metric = _metric_in(sent) or metric_used
            if a != b and a in models and b in models:
                sa, sb = models[a].get(cmp_metric), models[b].get(cmp_metric)
                if isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
                    claim = None
                    if any(w in low for w in PASSIVE_POSITIVE):
                        claim = (b, a)        # b is higher than a
                    elif any(w in low for w in POSITIVE):
                        claim = (a, b)
                    elif any(w in low for w in NEGATIVE):
                        claim = (b, a)
                    if claim:
                        hi, lo = claim
                        if models[hi][cmp_metric] <= models[lo][cmp_metric]:
                            findings.append(_finding(
                                "comparison", "high", sent,
                                f"claims {hi} beats {lo} on {cmp_metric}, but the scores are "
                                f"{models[hi][cmp_metric]:.4f} vs {models[lo][cmp_metric]:.4f}"))

        # 3b. ladder direction: "X improved the score to <s>" when that step actually lowered it
        if re.search(r"\b(improv|rais|lift|increas|boost|gain|add(ed|s)? \d)", low):
            ladder = [s for s in selection.get("ladder", []) if isinstance(s.get("score"), (int, float))]
            for i in range(1, len(ladder)):
                prev, cur = ladder[i - 1]["score"], ladder[i]["score"]
                if cur >= prev:
                    continue
                for m in NUMBER_RE.finditer(sent):
                    raw = m.group(1)
                    if "." in raw and m.group(2) != "%" and _matches(float(raw.replace(",", "")),
                                                                    len(raw.split(".")[1]), False, [cur]):
                        findings.append(_finding(
                            "ladder", "high", sent,
                            f"describes '{ladder[i].get('label', 'step')}' as an improvement, but it lowered "
                            f"the score from {prev:.4f} to {cur:.4f}"))
                        break

        # 4. winner claims
        if best and found and any(w in low for w in WINNER_WORDS) and "baseline" not in low \
                and not any(w in low for w in ("runner-up", "runner up", "second", "worst", "not the", "unlike")):
            if best not in keys:
                findings.append(_finding(
                    "winner", "high", sent,
                    f"names {sorted(keys)} as best/selected, but the winner was {best}"))

        # 5. baseline description
        if re.search(r"majority[- ]class|random(ly)? guess|always predict|dummy classifier", low):
            findings.append(_finding(
                "baseline", "high", sent,
                "the baseline is a fixed logistic regression on a fixed preprocessing recipe, "
                "not a majority-class or random guesser"))

        # 6. dropped columns
        # "clipped to remove extreme values" removes values, not columns, so outlier sentences are skipped
        if re.search(r"\b(dropp|remov|discard)", low) and not re.search(r"\b(not|never|no|none|without)\b", low) \
                and not re.search(r"\b(clip|outlier|extreme|cap|capped|winsor)", low):
            decisions = facts.get("preprocessing", {}).get("column_decisions", [])
            dropped = {d["column"] for d in decisions if d.get("missing_action") == "drop_column"}
            all_cols = {d["column"] for d in decisions}
            norm = re.sub(r"[‐‑‒–]", "-", low)      # LLMs write "education‑num" with U+2011
            for col in all_cols:
                if re.search(r"(?<![\w-])" + re.escape(col.lower()) + r"(?![\w-])", norm) and col not in dropped:
                    findings.append(_finding(
                        "column", "medium", sent, f"says '{col}' was dropped, but it was kept"))

    unique: List[Dict[str, str]] = []
    for f in findings:                       # the same number twice in one sentence is one finding
        if f not in unique:
            unique.append(f)
    findings = unique
    flagged = len(findings)
    return {
        "summary": {
            "sentences": len(sentences),
            "numbers_checked": checked,
            "numbers_verified": verified,
            "claims_flagged": flagged,
            "trust": "no claims flagged" if flagged == 0 else f"{flagged} claim(s) flagged",
        },
        "findings": findings,
    }
