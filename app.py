"""Streamlit demo: run the ranker on a small uploaded candidate sample."""

import json

import streamlit as st

from src import honeypot
from src.features import extract, behavioral_modifier
from src.io_utils import reference_day
from src.reasoning import build as build_reason
from src.scoring import score as score_fit

DEFAULT = "data/sample_candidates.json"

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob - Candidate Ranker")
st.caption("Rule-based, CPU-only. No LLM or network calls at ranking time.")

uploaded = st.file_uploader("Candidate sample (.json array or .jsonl)",
                            type=["json", "jsonl"])
top_k = st.slider("How many to rank", 5, 100, 20)


def _load(file):
    if file is None:
        with open(DEFAULT, "r", encoding="utf-8") as f:
            return json.load(f)
    raw = file.read().decode("utf-8")
    if file.name.endswith(".jsonl"):
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    return json.loads(raw)


if st.button("Rank", type="primary") or uploaded is None:
    cands = _load(uploaded)

    scored, max_active = [], ""
    for c in cands:
        feat = extract(c)
        max_active = max(max_active, feat["signals"].get("last_active_date") or "")
        is_hp, hp = honeypot.detect(feat)
        fit, strengths, concerns = score_fit(feat)
        scored.append((feat, fit, strengths, concerns, is_hp, hp))

    ref = reference_day(max_active)
    finals = []
    for feat, fit, strengths, concerns, is_hp, hp in scored:
        mult, notes = behavioral_modifier(feat["signals"], ref)
        final = fit * mult
        if is_hp:
            final *= 0.0005
            concerns = ["likely honeypot: " + "; ".join(hp[:2])] + concerns
        finals.append((feat, final, strengths, concerns, notes))

    finals.sort(key=lambda r: (-r[1], r[0]["candidate_id"]))
    finals = finals[:top_k]
    hi = finals[0][1] if finals else 1.0
    lo = finals[-1][1] if finals else 0.0
    span = (hi - lo) or 1.0

    table, csv_rows = [], ["candidate_id,rank,score,reasoning"]
    for i, (feat, raw, s, c, notes) in enumerate(finals):
        disp = round(0.05 + 0.94 * (raw - lo) / span, 4)
        reason = build_reason(feat, i + 1, disp, s, c, notes)
        table.append({"rank": i + 1, "candidate_id": feat["candidate_id"],
                      "title": feat["current_title"], "years": feat["years"],
                      "score": disp, "reasoning": reason})
        csv_rows.append(f'{feat["candidate_id"]},{i+1},{disp},"{reason}"')

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download submission.csv", "\n".join(csv_rows),
                       "submission.csv", "text/csv")
