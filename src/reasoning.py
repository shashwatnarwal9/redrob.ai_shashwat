"""Build a short, grounded reasoning line per candidate."""

_ACRONYMS = {" Ai": " AI", " Ml": " ML", " Nlp": " NLP", "Qa ": "QA ",
             " Hr ": " HR ", ".net": ".NET"}

# Templates per score band; "_C" variants are used when a concern exists.
_STRONG = [
    "Strong fit: {title}, with {strengths}.",
    "{title}; {strengths}.",
    "{title} - {strengths}.",
]
_STRONG_C = [
    "Strong fit: {title}, with {strengths}; {concern}.",
    "{title}; {strengths}, though {concern}.",
    "{title} - {strengths}, with one caveat: {concern}.",
]
_MID_C = [
    "{title}; {strengths}, though {concern}.",
    "{title} - partial fit on {strengths}, but {concern}.",
    "{title}: {strengths}; {concern}.",
]
_WEAK_C = [
    "{title}; some {strengths}, but {concern}.",
    "{title} - only partial: {strengths}; {concern}.",
    "{title}: {strengths}, however {concern}.",
]


def _title_phrase(feat):
    t = (feat["current_title"] or "professional").title()
    for bad, good in _ACRONYMS.items():
        t = t.replace(bad, good)
    return f"{t} with {feat['years']:.0f}y experience"


def _join(items, limit=2):
    items = [i for i in items if i][:limit]
    if not items:
        return ""
    return items[0] if len(items) == 1 else items[0] + " and " + items[1]


def build(feat, rank, score, strengths, concerns, behav_notes):
    v = sum(ord(c) for c in feat["candidate_id"]) % 3
    title = _title_phrase(feat)
    strong_txt = _join(strengths, 2) or "adjacent technical skills"
    all_concerns = list(concerns) + list(behav_notes)
    concern_txt = _join(all_concerns, 2)

    if score >= 0.5:
        tmpl = (_STRONG_C if all_concerns else _STRONG)[v]
    elif score >= 0.2:
        tmpl = _MID_C[v] if all_concerns else _STRONG[v]
    else:
        tmpl = _WEAK_C[v] if all_concerns else _MID_C[v]

    text = tmpl.format(title=title, strengths=strong_txt,
                       concern=concern_txt or "narrowly behind stronger profiles")
    text = " ".join(text.split())
    return (text if text.endswith(".") else text + ".").replace("\n", " ")
