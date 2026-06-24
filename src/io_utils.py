"""Load candidates (.jsonl / .jsonl.gz / .json) and write the submission CSV."""

import csv
import gzip
import json
from datetime import date


def load_candidates(path):
    opener = (lambda p: gzip.open(p, "rt", encoding="utf-8")) if path.endswith(".gz") \
        else (lambda p: open(p, "r", encoding="utf-8"))
    with opener(path) as f:
        if path.endswith(".json"):
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def reference_day(max_iso):
    # "Today" = the latest last_active_date in the pool (fallback fixed).
    if max_iso:
        try:
            y, m, d = (int(x) for x in max_iso.split("-"))
            return date(y, m, d)
        except (ValueError, TypeError):
            pass
    return date(2026, 6, 23)


def write_submission(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rank, sc, reason in rows:
            w.writerow([cid, rank, f"{sc:.4f}", reason])
