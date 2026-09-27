from app.services.generation import _call_llm, _extract_final_answer

FORMULATION_CATEGORIES = """
- Classical/generic medicine: formulation and method drawn from a First-Schedule authoritative Ayurvedic text. Faces the Section 3(p) patenting bar; defended via Traditional Knowledge Digital Library (TKDL), not patents.
- Patent-or-proprietary medicine: a branded formulation not identical to a classical text formula, but still largely combining known ingredients in known ways.
- New/non-classical drug: a genuinely novel formulation, process, or extraction requiring safety and efficacy evidence. Has real patent potential.
- Phytopharmaceutical: a purified/standardized botanical drug substance, regulated distinctly under Indian drug rules.
- Ayurveda-Aahar / nutraceutical: food-category product with a health claim, regulated by FSSAI, not as a drug.
- Cosmetic: topical product with no therapeutic claim.
"""

TKDL_INFO = {
    "url": "https://www.tkdl.res.in",
    "note": "Classical Ayurvedic formulations are documented in the Traditional Knowledge Digital Library (TKDL), used by patent offices worldwide as prior art to prevent misappropriation. TKDL is not publicly searchable, but its existence is grounds to challenge patents based on known classical formulations.",
}

def classify_formulation(description: str, regime: str) -> dict:
    prompt = f"""You are classifying an Ayurvedic product description into one of the following categories, for IP and regulatory purposes:

{FORMULATION_CATEGORIES}

Jurisdiction context: {regime}

Product description: {description}

Respond in EXACTLY this format:
CATEGORY: <one category name from the list above>
REASONING: <one or two sentences explaining why, based only on the description given>
IP_POSTURE: <one or two sentences on what this category generally means for patentability/protection in {regime}>
CLARIFYING_QUESTIONS: <up to 3 short questions, separated by " | ", that would help refine or confirm this classification if the description is ambiguous. If the description is already clear, write "None needed">
"""

    raw = _call_llm(prompt, max_tokens=600)
    raw = _extract_final_answer(raw)

    result = {
        "formulation_type": "Unclassified",
        "reasoning": raw,
        "ip_posture": "",
        "clarifying_questions": [],
    }

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            result["formulation_type"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("IP_POSTURE:"):
            result["ip_posture"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CLARIFYING_QUESTIONS:"):
            q_text = line.split(":", 1)[1].strip()
            if q_text.lower() != "none needed":
                result["clarifying_questions"] = [q.strip() for q in q_text.split("|") if q.strip()]

    return result
