"""Fit scoring: coherence gate, weighted signal blend, exclusion penalties."""

from src import jd_profile as J
from src.features import hits, relevant_skill_depth, applied_ml_months

RELEVANT_SKILL_VOCAB = J.RETRIEVAL_RANKING | J.EMBEDDINGS | J.VECTOR_DB | J.ML_CORE
ML_EVIDENCE_VOCAB = J.ML_CORE | J.RETRIEVAL_RANKING | J.EMBEDDINGS
CODE_EVIDENCE_VOCAB = J.ML_CORE | {
    "python", "code", "built", "shipped", "deployed", "pipeline", "api",
    "production", "service", "backend",
}


def _sat(n, k):
    # Saturating 0..1 from a raw hit count.
    return 0.0 if n <= 0 else 1.0 - pow(2.718281828, -n / k)


def coherence(feat):
    """Is this an engineer or a non-tech profile with AI keywords? -> 0.12..1.0."""
    ident, evidence, title = feat["identity_text"], feat["evidence_text"], feat["current_title"]

    nontech_title = any(t in title for t in J.NON_ENGINEER_TITLE_TERMS) and \
        not any(t in title for t in (
            "ml engineer", "ai engineer", "software engineer", "data engineer",
            "data scientist", "research", "ml ", "machine learning",
        ))
    eng_title = any(t in title for t in J.ENGINEER_TITLE_TERMS)

    trap = hits(ident, J.TRAP_SELF_ID_PHRASES)
    eng = hits(ident, J.ENGINEER_SELF_ID_PHRASES)
    evidence_eng = hits(evidence, J.ML_CORE | J.RETRIEVAL_RANKING |
                        J.EMBEDDINGS | J.VECTOR_DB | {"python", "api", "backend",
                        "data pipeline", "kafka", "spark", "microservices",
                        "deployment", "kubernetes", "production"})

    c = 0.5
    if eng_title:
        c += 0.25
    if nontech_title:
        c -= 0.35
    c += 0.20 * _sat(eng, 2.0)
    c -= 0.30 * _sat(trap, 2.0)
    c *= 0.45 + 0.55 * _sat(evidence_eng, 3.0)   # claims must be backed by work
    return max(0.12, min(1.0, c))


def product_score(feat):
    """Product-company vs services exposure; returns (score, has_strong_product)."""
    history = feat["history"]
    if not history:
        return 0.4, False
    product_m = services_m = 0.0
    strong_product = False
    for h, comp, ind in zip(history, feat["companies"], feat["industries"]):
        dur = h.get("duration_months") or 0
        is_consult = any(cf in comp for cf in J.CONSULTING_FIRMS)
        is_prod_co = any(pc in comp for pc in J.PRODUCT_COMPANIES)
        if is_prod_co or (ind in J.PRODUCT_INDUSTRIES and not is_consult):
            product_m += dur
            strong_product = strong_product or is_prod_co
        elif is_consult or ind in J.SERVICES_INDUSTRIES:
            services_m += dur
        else:
            product_m += dur * 0.5
            services_m += dur * 0.5
    total = product_m + services_m
    ratio = product_m / total if total else 0.4
    return min(1.0, ratio + (0.15 if strong_product else 0.0)), strong_product


def experience_band(years):
    if 6 <= years <= 8:
        return 1.0
    if 5 <= years < 6 or 8 < years <= 9:
        return 0.85
    if 4 <= years < 5 or 9 < years <= 11:
        return 0.6
    if 2 <= years < 4 or 11 < years <= 14:
        return 0.35
    return 0.18


def location_score(feat):
    loc, country = feat["location"], feat["country"]
    if any(c in loc for c in J.PREFERRED_CITIES):
        return 1.0
    if any(c in loc for c in J.WELCOME_CITIES):
        return 0.9
    return 0.7 if "india" in country else 0.2


def score(feat):
    ev, sk, full = feat["evidence_text"], feat["skills_text"], feat["full_text"]

    # Evidence weighted above the (gameable) skills list throughout.
    retr = _sat(hits(ev, J.RETRIEVAL_RANKING) + 0.5 * hits(sk, J.RETRIEVAL_RANKING), 2.0)
    emb = _sat(hits(full, J.EMBEDDINGS), 1.5)
    vdb = _sat(hits(full, J.VECTOR_DB), 1.0)
    mldepth = _sat(hits(ev, J.ML_CORE) + 0.5 * hits(sk, J.ML_CORE), 3.0)
    evalx = _sat(hits(ev, J.EVAL), 1.5)
    prod, strong_product = product_score(feat)
    expb = experience_band(feat["years"])
    locs = location_score(feat)
    skill_depth = _sat(relevant_skill_depth(feat, RELEVANT_SKILL_VOCAB), 2.5)
    ml_years = applied_ml_months(feat, J.ML_ROLE_TERMS, ML_EVIDENCE_VOCAB) / 12.0
    ml_tenure = min(1.0, ml_years / 4.5)

    retrieval_ranking = max(retr, 0.55 * emb)

    blend = (
        J.FIT_WEIGHTS["retrieval_ranking"] * retrieval_ranking +
        J.FIT_WEIGHTS["ml_depth"] * mldepth +
        J.FIT_WEIGHTS["vector_db"] * vdb +
        J.FIT_WEIGHTS["evaluation"] * evalx +
        J.FIT_WEIGHTS["product_company"] * prod +
        J.FIT_WEIGHTS["experience_band"] * expb +
        J.FIT_WEIGHTS["ml_tenure"] * ml_tenure +
        J.FIT_WEIGHTS["skill_depth"] * skill_depth +
        J.FIT_WEIGHTS["location"] * locs
    )

    coh = coherence(feat)
    fit = coh * blend

    concerns, strengths = [], []
    penalty = 1.0

    cv = hits(full, J.CV_SPEECH_ROBOTICS)
    nlp_ir = hits(full, J.RETRIEVAL_RANKING | J.EMBEDDINGS |
                  {"nlp", "natural language processing", "information retrieval"})
    if cv >= 3 and nlp_ir <= 1:
        penalty *= 0.75
        concerns.append("primarily CV/speech, thin NLP/IR")

    all_consulting = feat["companies"] and all(
        any(cf in c for cf in J.CONSULTING_FIRMS) for c in feat["companies"])
    if all_consulting and not strong_product:
        penalty *= 0.78
        concerns.append("entire career at services/consulting firms")

    if hits(full, J.LLM_GENAI) >= 2 and mldepth < 0.3 and retr < 0.3:
        penalty *= 0.85
        concerns.append("recent GenAI keywords without pre-LLM ML depth")

    short_stints = sum(1 for h in feat["history"]
                       if (h.get("duration_months") or 99) < 20)
    if short_stints >= 3 and len(feat["history"]) >= 4:
        penalty *= 0.92
        concerns.append("frequent short tenures")

    # Management/architecture title with no recent hands-on code evidence.
    if any(t in feat["current_title"] for t in J.NON_CODING_LEADER_TERMS) and \
            hits(ev, CODE_EVIDENCE_VOCAB) < 2:
        penalty *= 0.85
        concerns.append("in management/architecture, light on hands-on code")

    if "india" not in feat["country"]:
        concerns.append(f"based outside India ({feat['country'] or 'unknown'})")

    fit = max(0.001, min(1.0, fit * penalty))

    # Strengths, grounded only in signals that actually fired.
    if retr > 0.5:
        strengths.append("ranking/retrieval/search experience")
    if vdb > 0.4:
        present = [v for v in ("faiss", "pinecone", "qdrant", "weaviate",
                   "milvus", "opensearch", "elasticsearch") if v in full]
        if present:
            strengths.append("vector search (" +
                             "/".join(p.capitalize() for p in present[:3]) + ")")
    if emb > 0.4:
        strengths.append("embeddings-based retrieval")
    if evalx > 0.4:
        metrics = [m.upper() for m in ("ndcg", "mrr") if m in full]
        if metrics:
            strengths.append("evaluation rigor (" + "/".join(metrics) + ")")
        elif "a/b" in full or "ab test" in full or "experiment" in full:
            strengths.append("A/B-testing & evaluation experience")
        else:
            strengths.append("ranking-evaluation experience")
    if strong_product:
        names = [c for c in feat["companies"]
                 if any(pc in c for pc in J.PRODUCT_COMPANIES)]
        if names:
            strengths.append("product-company background (" + names[0].title() + ")")
    if ml_years >= 4 and coh >= 0.5:
        strengths.append(f"~{ml_years:.0f}y in applied-ML roles")
    if 5 <= feat["years"] <= 9:
        strengths.append(f"{feat['years']:.0f}y in the JD's 5-9 band")

    if coh < 0.4:
        concerns.append("profile reads as non-engineer / keyword-stuffed")
    if not strengths:
        strengths.append("adjacent technical skills")

    # Honest caveat from the weakest signal so no row gets a fabricated concern.
    if not concerns:
        if expb < 0.4:
            concerns.append(f"{feat['years']:.0f}y sits outside the JD's 5-9 band")
        elif prod < 0.45:
            concerns.append("limited product-company signal in the career history")
        elif ml_tenure < 0.45:
            concerns.append("applied-ML tenure on the lighter side")
        elif evalx < 0.3:
            concerns.append("little explicit ranking-evaluation (NDCG/A-B) signal")
        elif vdb < 0.3:
            concerns.append("thin vector-DB / hybrid-search exposure")

    return fit, strengths, concerns
