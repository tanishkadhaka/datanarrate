"""
PDF report export
Builds a one-click PDF of a finished run from the API's result payload (a dict).

    pdf_bytes = build_pdf(payload, dataset_name="adult.csv")
"""
import io
from datetime import datetime
from typing import Any, Dict, List
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from reportlab.lib import colors                                   # noqa: E402
from reportlab.lib.enums import TA_LEFT                            # noqa: E402
from reportlab.lib.pagesizes import A4                             # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm                                 # noqa: E402
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,   # noqa: E402
                                Spacer, Table, TableStyle)

PURPLE = colors.HexColor("#6C5CE7")
TEAL = colors.HexColor("#00A8A9")
GREY = colors.HexColor("#6B7280")
LIGHT = colors.HexColor("#F3F4F8")

ACTION = {"none": "Nothing missing", "impute_median": "Median", "impute_mode": "Most frequent value",
          "impute_constant": "'Missing' category", "drop_column": "Dropped"}


def _ascii(text: Any) -> str:
    """The built-in PDF fonts cannot draw every Unicode character, so swap the common ones."""
    s = str(text if text is not None else "")
    for a, b in {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                 "‐": "-", "‑": "-", "‒": "-", "−": "-",          # hyphens / minus
                 " ": " ", " ": " ", " ": " ", " ": " ",         # thin / narrow spaces
                 "​": "",
                 "→": "->", "≥": ">=", "≤": "<=", "×": "x", "…": "...",
                 "✅": "", "⚠": "", "\U0001F9E0": "", "\U0001F3C6": ""}.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def _p(text: Any, style) -> Paragraph:
    return Paragraph(escape(_ascii(text)), style)


def _num(x: Any, d: int = 4) -> str:
    return "-" if x is None or x != x else f"{float(x):.{d}f}"


def _table(rows: List[List[Any]], col_widths, header=True, font=8) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("LINEBELOW", (0, 0), (-1, 0), 0.8, PURPLE), ("TOPPADDING", (0, 0), (-1, -1), 3),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), PURPLE)]
    t.setStyle(TableStyle(style))
    return t


def _ladder_chart(ladder: List[Dict[str, Any]], metric: str) -> Image:
    labels = [l["label"].replace("+ ", "+\n") for l in ladder]
    scores = [l["score"] if l["score"] is not None else 0 for l in ladder]
    lo = max(0.0, min(scores) - 0.05)
    hi = min(1.0, max(scores) + 0.03)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=170)
    bars = ax.bar(labels, scores, color=["#9CA3AF", "#8E7CF0", "#6C5CE7", "#00A8A9"][: len(scores)])
    ax.set_ylim(lo, hi)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s, f"{s:.4f}", ha="center", va="bottom", fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=15.5 * cm, height=6.3 * cm)


def build_pdf(payload: Dict[str, Any], dataset_name: str = "") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="DataNarrate report", author="DataNarrate")
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, textColor=PURPLE, alignment=TA_LEFT, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=PURPLE, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=GREY)
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)

    sel = payload.get("selection", {})
    miss = payload.get("missing_report", {})
    metric = sel.get("metric_used", "score")
    imp = sel.get("improvement_over_baseline") or 0
    story: List[Any] = []

    story += [_p("DataNarrate report", title),
              _p(f"Dataset: {dataset_name or payload.get('dataset_id', '-')}   |   Target: {payload.get('target_column', '-')}"
                 f"   |   Generated {datetime.now():%Y-%m-%d %H:%M}", small), Spacer(1, 8)]

    banner = Table([[_p(f"Winner: {sel.get('best_model_name', '-')}   |   {metric}: {_num(sel.get('best_score'))}   |   "
                        f"baseline: {_num(sel.get('baseline_score'))}   |   change: {imp:+.4f}",
                        ParagraphStyle("w", parent=body, textColor=colors.white, fontSize=10, fontName="Helvetica-Bold"))]],
                   colWidths=[17.4 * cm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PURPLE), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [banner, Spacer(1, 6), _p(sel.get("final_justification", ""), body)]

    # --- missing data
    story.append(_p("1. Missing data", h2))
    story.append(_p(" ".join(miss.get("summary", [])), body))
    cols = [c for c in miss.get("columns", []) if c.get("Total missing", 0) > 0]
    cols.sort(key=lambda c: -c.get("Missing %", 0))
    if cols:
        rows = [["Column", "Type", "Empty", "Hidden", "Missing %", "Severity", "Placeholders found"]]
        for c in cols[:15]:
            rows.append([_p(c["Column"], cell), c["Type"], c["Empty cells"], c["Hidden placeholders"],
                         f'{c["Missing %"]}%', c["Severity"], _p(c.get("Placeholders found", ""), cell)])
        story += [Spacer(1, 4), _table(rows, [3.6 * cm, 1.6 * cm, 1.5 * cm, 1.6 * cm, 2 * cm, 2 * cm, 5 * cm])]
    for w in miss.get("warnings", [])[:5]:
        story.append(_p("Note: " + w, small))

    mh = payload.get("missing_handling", [])
    if mh:
        story.append(_p("How missing values were handled", h2))
        rows = [["Column", "Missing %", "Decision", "'Was missing' flag", "Rule"]]
        for r in sorted(mh, key=lambda r: -r["missing_pct"])[:20]:
            rows.append([_p(r["column"], cell), f'{r["missing_pct"]}%', ACTION.get(r["action"], r["action"]),
                         "Yes" if r["was_missing_flag"] else "No", r["rule"]])
        story.append(_table(rows, [4.6 * cm, 2.2 * cm, 4.6 * cm, 3.4 * cm, 2.2 * cm]))

    # --- preprocessing
    story.append(_p("2. Preprocessing decisions", h2))
    for line in payload.get("preprocessing", {}).get("summary", []):
        story.append(_p("- " + line, body))
    for line in payload.get("target_notes", []):
        story.append(_p("- Target: " + line, body))

    # --- models
    story.append(_p("3. Model comparison (cross-validated)", h2))
    rows = [["Model", "Family", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    for m in sorted(sel.get("candidate_results", []), key=lambda m: -(m.get(metric) or 0)):
        name = m["model_name"] + (" *" if m["model_name"] == sel.get("best_model_name") else "")
        rows.append([name, _p(m.get("family", ""), cell), _num(m["accuracy"]), _num(m["precision"]),
                     _num(m["recall"]), _num(m["f1"]), _num(m.get("roc_auc"))])
    story += [_table(rows, [3.4 * cm, 4.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm]),
              _p(f"* best model by {metric}. Cross-validation folds: {sel.get('cv_folds_used', '-')}.", small)]
    for k, v in (sel.get("skipped_models") or {}).items():
        story.append(_p(f"{k}: {v}", small))

    # --- ladder
    ladder = sel.get("ladder") or []
    if ladder:
        story.append(_p("4. Where the improvement comes from", h2))
        story.append(KeepTogether([_ladder_chart(ladder, metric)]))
        rows = [["Step", metric, "Change", "What changed"]]
        prev = None
        for l in ladder:
            ch = "-" if prev is None or l["score"] is None else f'{l["score"] - prev:+.4f}'
            rows.append([l["label"], _num(l["score"]), ch, _p(l["what_changed"], cell)])
            prev = l["score"]
        story.append(_table(rows, [4.8 * cm, 1.8 * cm, 1.8 * cm, 9.0 * cm]))
    info = sel.get("baseline_info")
    if info:
        story += [Spacer(1, 4), _p("Baseline: " + info.get("summary", ""), small)]

    # --- narration + fact check
    if payload.get("narration"):
        story.append(_p("5. Narration", h2))
        for para in str(payload["narration"]).split("\n\n"):
            story.append(_p(para, body))
        fc = payload.get("fact_check")
        if fc:
            s = fc["summary"]
            story.append(_p(f"Fact-check: {s['numbers_verified']} of {s['numbers_checked']} numbers verified; "
                            f"{s['claims_flagged']} claim(s) flagged.", small))
            for f in fc["findings"][:8]:
                story.append(_p(f"- [{f['type']}] {f['detail']}", small))

    story += [Spacer(1, 14), _p("Generated by DataNarrate. All numbers come from cross-validated runs; "
                                 "preprocessing was fit inside each fold.", small)]
    doc.build(story)
    return buf.getvalue()
