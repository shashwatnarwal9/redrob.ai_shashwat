"""Feature extraction and keyword matching for a candidate record."""

import re
from datetime import date

_matcher_cache = {}
_PROF_W = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


def _norm(s):
    return (s or "").lower().strip()


def _build_matcher(vocab):
    # Single tokens matched with a leading boundary + open stem so "rank" hits
    # ranking/ranked but not "frank"; multi-word phrases use plain containment.
    multiword, single = [], []
    for t in vocab:
        (single if t.isalnum() else multiword).append(t)
    rx = None
    if single:
        alt = "|".join(sorted((re.escape(t) for t in single),
                              key=len, reverse=True))
        rx = re.compile(r"(?<![a-z0-9])(?:" + alt + r")[a-z0-9]*")
    return multiword, rx


def hits(text, vocab):
    """Count distinct vocab phrases present in text."""
    entry = _matcher_cache.get(id(vocab))
    if entry is None:
        entry = _build_matcher(vocab)
        _matcher_cache[id(vocab)] = entry
    multiword, rx = entry
    n = len(set(rx.findall(text))) if rx else 0
    return n + sum(1 for t in multiword if t in text)


def extract(cand):
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []

    summary = _norm(profile.get("summary"))
    current_title = _norm(profile.get("current_title"))
    headline = _norm(profile.get("headline"))
    skills_text = " ".join(_norm(s.get("name")) for s in cand.get("skills") or [])
    desc_text = " ".join(_norm(h.get("description")) for h in history)

    # Three views: what they claim, what they did, and the gameable skills list.
    identity_text = f"{headline} {current_title} {summary}"
    evidence_text = f"{summary} {desc_text}"
    full_text = f"{identity_text} {desc_text} {skills_text}"

    return {
        "candidate_id": cand.get("candidate_id", ""),
        "current_title": current_title,
        "years": float(profile.get("years_of_experience", 0) or 0),
        "location": _norm(profile.get("location")),
        "country": _norm(profile.get("country")),
        "identity_text": identity_text,
        "evidence_text": evidence_text,
        "skills_text": skills_text,
        "full_text": full_text,
        "skills": cand.get("skills") or [],
        "history": history,
        "companies": [_norm(h.get("company")) for h in history],
        "industries": [_norm(h.get("industry")) for h in history],
        "education": cand.get("education") or [],
        "signals": cand.get("redrob_signals") or {},
    }


def behavioral_modifier(sig, reference_day):
    """Availability multiplier (~0.45..1.10) from activity / response signals."""
    if not sig:
        return 0.8, []

    notes, mult = [], 1.0

    last = sig.get("last_active_date")
    days_idle = None
    if last:
        try:
            y, m, d = (int(x) for x in last.split("-"))
            days_idle = (reference_day - date(y, m, d)).days
        except (ValueError, TypeError):
            days_idle = None
    if days_idle is not None:
        if days_idle <= 45:
            mult *= 1.05
        elif days_idle <= 120:
            mult *= 0.97
        elif days_idle <= 210:
            mult *= 0.85
            notes.append(f"inactive ~{days_idle // 30}mo")
        else:
            mult *= 0.65
            notes.append(f"dormant ~{days_idle // 30}mo")

    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        if rr < 0.10:
            mult *= 0.60
            notes.append(f"very low response rate {rr:.0%}")
        elif rr < 0.25:
            mult *= 0.82
            notes.append(f"low response rate {rr:.0%}")
        elif rr >= 0.60:
            mult *= 1.05

    mult *= 1.04 if sig.get("open_to_work_flag") else 0.96

    icr = sig.get("interview_completion_rate")
    if icr is not None and icr < 0.4:
        mult *= 0.93
    pcs = sig.get("profile_completeness_score")
    if pcs is not None and pcs < 35:
        mult *= 0.95
        notes.append("sparse profile")

    npd = sig.get("notice_period_days")
    if npd is not None and npd >= 120:
        mult *= 0.97
        notes.append(f"{npd}d notice")

    return max(0.45, min(1.10, mult)), notes


def _skill_trust(skill, assessment):
    # Proficiency tempered by endorsements, months used, and assessment score.
    t = _PROF_W.get((skill.get("proficiency") or "").lower(), 0.4)
    t *= 0.4 + 0.6 * min(skill.get("endorsements") or 0, 20) / 20.0
    t *= 0.4 + 0.6 * min(skill.get("duration_months") or 0, 36) / 36.0
    if assessment is not None:
        t *= 0.5 + 0.5 * (assessment / 100.0)
    return t


def relevant_skill_depth(feat, vocab):
    """Summed trust over skills whose name matches the relevant vocab."""
    assess = {str(k).lower(): v for k, v in
              (feat["signals"].get("skill_assessment_scores") or {}).items()}
    total = 0.0
    for s in feat["skills"]:
        name = _norm(s.get("name"))
        if name and hits(name, vocab):
            total += _skill_trust(s, assess.get(name))
    return total


def applied_ml_months(feat, ml_role_terms, ml_vocab):
    """Months in roles that are an ML title or whose description shows ML work."""
    months = 0
    for h in feat["history"]:
        title = _norm(h.get("title"))
        if any(t in title for t in ml_role_terms) or \
                hits(_norm(h.get("description")), ml_vocab) >= 2:
            months += h.get("duration_months") or 0
    return months
