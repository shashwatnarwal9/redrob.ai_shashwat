
import json
import time

import streamlit as st

from src import honeypot
from src.features import extract, behavioral_modifier
from src.io_utils import reference_day
from src.reasoning import build as build_reason
from src.scoring import score as score_fit

DEFAULT = "data/sample_candidates.json"

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯",
                   layout="wide")


def load(file):
    if file is None:
        with open(DEFAULT, "r", encoding="utf-8") as f:
            return json.load(f)
    raw = file.read().decode("utf-8")
    if file.name.endswith(".jsonl"):
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    return json.loads(raw)


def rank_candidates(cands, top_k):
    scored, max_active, honeypots = [], "", 0
    for c in cands:
        feat = extract(c)
        max_active = max(max_active, feat["signals"].get("last_active_date") or "")
        is_hp, hp = honeypot.detect(feat)
        honeypots += is_hp
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
        table.append({"Rank": i + 1, "Candidate": feat["candidate_id"],
                      "Title": (feat["current_title"] or "").title(),
                      "Years": round(feat["years"], 1), "Fit": disp,
                      "Reasoning": reason})
        csv_rows.append(f'{feat["candidate_id"]},{i+1},{disp},"{reason}"')
    return table, "\n".join(csv_rows), honeypots


# ---- header -----------------------------------------------------------------
st.title("🎯 Redrob Candidate Ranker")
st.markdown(
    "Ranks candidates for the **Senior AI Engineer** JD by reading the gap "
    "between profile claims and real career evidence — coherence-gated, "
    "CPU-only, **no LLM or network calls**."
)

# ---- sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Rank your candidates")
    uploaded = st.file_uploader("Candidate file (.json / .jsonl, up to ~100)",
                                type=["json", "jsonl"])
    top_k = st.slider("How many to show", 5, 100, 20)
    go = st.button("Rank", type="primary", use_container_width=True)
    st.caption("No file? Just click **Rank** to use the bundled 50-candidate "
               "sample.")
    with st.expander("Expected input format"):
        st.markdown(
            "A `.json` array or `.jsonl` of candidate records matching "
            "`candidate_schema.json` (same shape as `sample_candidates.json`): "
            "`candidate_id`, `profile`, `career_history`, `education`, `skills`, "
            "`redrob_signals`."
        )

# ---- body -------------------------------------------------------------------
if go:
    t0 = time.time()
    cands = load(uploaded)
    table, csv_text, honeypots = rank_candidates(cands, top_k)
    elapsed = (time.time() - t0) * 1000

    a, b, c, d = st.columns(4)
    a.metric("Candidates scored", len(cands))
    b.metric("Shortlist shown", len(table))
    c.metric("Honeypots filtered", honeypots)
    d.metric("Runtime", f"{elapsed:.0f} ms")

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Candidate": st.column_config.TextColumn(width="medium"),
            "Years": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Fit": st.column_config.ProgressColumn(
                "Fit score", min_value=0.0, max_value=1.0, format="%.2f"),
            "Reasoning": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button("⬇️ Download submission.csv", csv_text,
                       "submission.csv", "text/csv")

    st.subheader("Reasoning")
    st.caption("Full text for each shortlisted candidate (table cells truncate).")
    for row in table:
        st.markdown(
            f"**#{row['Rank']} · {row['Candidate']}** "
            f"· {row['Title']} · {row['Years']}y · fit {row['Fit']:.2f}  \n"
            f"{row['Reasoning']}"
        )
else:
    st.info("Use the sidebar — upload a candidate file (or just click **Rank**) "
            "to score and shortlist candidates.")
