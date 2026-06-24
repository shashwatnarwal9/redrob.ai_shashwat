#!/usr/bin/env python3
"""
Rank candidates for the Redrob "Senior AI Engineer" JD.

    python rank.py --candidates ./candidates.jsonl.gz --out ./CNMD.csv

CPU-only, no network, Python stdlib only.
"""

import argparse
import time

from src import honeypot
from src.features import extract, behavioral_modifier
from src.io_utils import load_candidates, reference_day, write_submission
from src.reasoning import build as build_reason
from src.scoring import score as score_fit

TOP_N = 100


def run(candidates_path, out_path):
    t0 = time.time()

    # Pass 1: features + fit score. Track the latest activity date as "today".
    scored, max_active, n = [], "", 0
    for cand in load_candidates(candidates_path):
        feat = extract(cand)
        n += 1
        la = feat["signals"].get("last_active_date") or ""
        if la > max_active:
            max_active = la
        is_hp, hp_reasons = honeypot.detect(feat)
        fit, strengths, concerns = score_fit(feat)
        scored.append((feat, fit, strengths, concerns, is_hp, hp_reasons))

    ref_day = reference_day(max_active)

    # Pass 2: behavioral multiplier + honeypot floor.
    finals = []
    for feat, fit, strengths, concerns, is_hp, hp_reasons in scored:
        mult, behav_notes = behavioral_modifier(feat["signals"], ref_day)
        final = fit * mult
        if is_hp:
            final *= 0.0005
            concerns = ["likely honeypot: " + "; ".join(hp_reasons[:2])] + concerns
        finals.append((feat, final, strengths, concerns, behav_notes))

    finals.sort(key=lambda r: (-r[1], r[0]["candidate_id"]))
    top = finals[:TOP_N]

    # Display score = min-max normalized fit; re-sort so equal rounded scores
    # break ties by candidate_id ascending (validator requirement).
    raws = [r[1] for r in top]
    hi, lo = (raws[0], raws[-1]) if raws else (1.0, 0.0)
    span = (hi - lo) or 1.0
    prelim = []
    for feat, raw, strengths, concerns, behav_notes in top:
        disp = round(0.05 + 0.94 * (raw - lo) / span, 4)
        prelim.append((disp, feat["candidate_id"], feat, strengths, concerns, behav_notes))
    prelim.sort(key=lambda r: (-r[0], r[1]))

    rows = []
    for i, (disp, cid, feat, strengths, concerns, behav_notes) in enumerate(prelim):
        reason = build_reason(feat, i + 1, disp, strengths, concerns, behav_notes)
        rows.append((cid, i + 1, disp, reason))

    write_submission(rows, out_path)
    print(f"Ranked {n} candidates -> top {len(rows)} -> {out_path} "
          f"({time.time() - t0:.1f}s)")
    for cid, rank, sc, reason in rows[:5]:
        print(f"  #{rank} {cid} {sc} | {reason[:90]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/sample_candidates.json")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    run(args.candidates, args.out)
