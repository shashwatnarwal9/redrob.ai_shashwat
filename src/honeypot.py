"""Detect internally-impossible profiles (forced to tier 0 in the ground truth)."""


def detect(feat):
    reasons = []
    months_total = feat["years"] * 12.0
    skills = feat["skills"]
    history = feat["history"]

    # Skill used longer than the whole career (+1yr grace).
    bad_span = sum(1 for s in skills
                   if months_total > 0 and (s.get("duration_months") or 0) > months_total + 12)
    if bad_span >= 2:
        reasons.append(f"{bad_span} skills used longer than total career")

    # "Expert"/"advanced" with zero months of usage.
    hollow = sum(1 for s in skills
                 if (s.get("proficiency") or "").lower() in ("expert", "advanced")
                 and (s.get("duration_months") or 0) == 0)
    if hollow >= 3:
        reasons.append(f"{hollow} expert skills with 0 months used")

    # A single role longer than the entire stated career.
    if any(months_total > 0 and (h.get("duration_months") or 0) > months_total + 6
           for h in history):
        reasons.append("a single role outlasts the whole career")

    # Career-history months far exceed stated experience.
    summed = sum((h.get("duration_months") or 0) for h in history)
    if months_total > 0 and summed > months_total * 1.8 and summed - months_total > 36:
        reasons.append("career-history duration far exceeds stated experience")

    # Education that ends before it begins.
    for e in feat["education"]:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            reasons.append("education ends before it begins")
            break

    # Require two independent impossibilities to avoid false positives.
    return len(reasons) >= 2, reasons
