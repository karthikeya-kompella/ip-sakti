# app/services/generation.py
import re
from openai import OpenAI
from app.config import settings

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

# Known script ranges — used only to sanity-check known languages when the
# user explicitly selects one. Auto-detect mode skips this check since we
# can't validate arbitrary/unknown scripts.
SCRIPT_RANGES = {
    "te": r'[\u0C00-\u0C7F]',
    "hi": r'[\u0900-\u097F]',
    "zh": r'[\u4E00-\u9FFF]',
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "zh": "Chinese",
}


def _call_llm(prompt: str, max_tokens: int = 1800) -> str:
    print(f"Calling LLM ({settings.GENERATION_MODEL_NAME})...")
    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL_NAME,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _extract_final_answer(text: str) -> str:
    """If the model leaked its own chain-of-thought before the real answer,
    take only the last substantial paragraph block, which is consistently
    where the actual answer lands."""
    if not text:
        return text

    markers = ["final answer:", "let's craft", "thus final answer:", "final answer"]
    lower = text.lower()
    cut_index = -1
    for marker in markers:
        idx = lower.rfind(marker)
        if idx > cut_index:
            cut_index = idx + len(marker)

    if cut_index > -1:
        candidate = text[cut_index:].strip(" :\n")
        if len(candidate) > 40:
            return candidate

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    if paragraphs:
        return max(paragraphs, key=len)

    return text.strip()


def _has_foreign_script_leak(text: str, target_code: str) -> bool:
    """Only meaningful when the user explicitly picked a known language.
    Skipped entirely in auto-detect mode."""
    for code, pattern in SCRIPT_RANGES.items():
        if code != target_code and re.search(pattern, text):
            return True
    return False
def check_ambiguity(question: str) -> dict:
    """Before running full retrieval + generation, check if the question is
    too vague to answer meaningfully. If so, return clarifying questions
    instead of guessing."""
    prompt = f"""You are a regulatory/IP assistant intake step. A user has asked the following question about an Ayurveda product's IP or regulatory status:

"{question}"

Decide if this question has ENOUGH detail to answer meaningfully, or if it is too vague.
A vague question is missing things like: what the formulation/ingredients are, what form it is (tablet, oil, powder), what claim or use it makes, or what specifically the user wants to know (patent? approval? classification?).

Respond in EXACTLY this format:
CLEAR: <yes or no>
QUESTIONS: <if CLEAR is no, list up to 3 short clarifying questions separated by " | " that would help. If CLEAR is yes, write "None">
"""

    raw = _call_llm(prompt, max_tokens=300)
    raw = _extract_final_answer(raw)

    is_clear = True
    questions = []

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("CLEAR:"):
            is_clear = "yes" in line.lower()
        elif line.upper().startswith("QUESTIONS:"):
            q_text = line.split(":", 1)[1].strip()
            if q_text.lower() != "none":
                questions = [q.strip() for q in q_text.split("|") if q.strip()]

    return {"is_clear": is_clear, "clarifying_questions": questions}

def generate_answer(
    question: str,
    chunks: list[dict],
    response_language: str | None = None,
) -> str:

    # response_language is optional. When it's None/"auto", the model
    # detects the question's own language and must match it — any
    # language, not a fixed list. When explicitly set to a known code,
    # that language is enforced and script-checked.
    explicit_lang = LANGUAGE_NAMES.get(response_language) if response_language else None

    if explicit_lang:
        language_instruction = f"""LANGUAGE RULE — THIS IS ABSOLUTE AND MANDATORY:
Write your ENTIRE answer in {explicit_lang} ONLY. Do not mix in any other language.
Proper nouns, Act names, and section numbers may stay in their original form, but every surrounding sentence must be in {explicit_lang}.
This rule overrides all other formatting preferences."""
    else:
        language_instruction = """LANGUAGE RULE — THIS IS ABSOLUTE AND MANDATORY:
The USER QUESTION below is written in a specific language. You MUST write your entire answer in that exact same language.
Example: if the question is in Telugu script, your entire answer must be in Telugu script. If in Hindi, answer in Hindi. If in English, answer in English.
DO NOT translate the question to English first and then answer in English. DO NOT default to English under any circumstance unless the question itself is literally written in English.
This rule overrides all formatting preferences — matching the question's language comes first, before anything else in this prompt.
Proper nouns, Act names, and section numbers may stay in their original script/form."""

    if not chunks:
        return "The available documents do not contain enough information to answer this question."

    context_parts = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title", "Unknown source")
        chunk_text = c.get("chunk_text", "")
        context_parts.append(f"SOURCE {i}\nTITLE: {title}\n\nCONTENT:\n{chunk_text}\n")

    context = "\n".join(context_parts)

    prompt = f"""{language_instruction}

You are IP-Sakti Sahayak, a jurisdiction-aware legal and regulatory assistant for Ayurveda IP and drug regulation.

CONTENT AND STRUCTURE RULES:
1. Base your answer ONLY on the retrieved sources below. Do not invent facts, sections, or authorities not present in the sources.
2. Structure your answer in three parts, clearly:
   a) DIRECT ANSWER — answer the user's question in one or two sentences.
   b) BASIS / REASONING — explain WHY this is the answer, by pointing to the specific rule, section, or passage in the sources that supports it. This is grounded justification, not your internal thought process.
   c) PRACTICAL GUIDANCE — what a person/company would need to DO to comply or proceed: which approval to obtain, which form to file, which authority to approach, what formulation/manufacturing condition applies, or what steps follow from the rule. If the sources describe a process, list it as ordered steps.
3. If the sources are insufficient to answer or to give practical guidance, say so explicitly and do not guess.
4. Do NOT include your own internal thinking, meta-commentary, or labels like "Let me think" or "Analysis:". Only the three sections above (a, b, c) should appear, each clearly, in the required language.

RETRIEVED SOURCES:
{context}

USER QUESTION:
{question}

FINAL ANSWER:"""

    try:
        raw_answer = _call_llm(prompt, max_tokens=1800)
        print(f"RAW ANSWER LENGTH: {len(raw_answer)}")
        print(f"RAW ANSWER PREVIEW: {raw_answer[:300]}")
        answer = _extract_final_answer(raw_answer)
        print(f"AFTER EXTRACTION LENGTH: {len(answer)}")
    except Exception as e:
        print(f"Generation error: {repr(e)}")
        answer = ""

    if not answer:
        return (
            f"(Model unavailable — showing top retrieved source directly.) "
            f"[{chunks[0].get('title','Source')}] {chunks[0].get('chunk_text','')[:500]}"
        )

    # Only sanity-check script purity when the user explicitly picked a
    # known language — auto-detect mode can't be validated this way.
    if response_language and response_language in SCRIPT_RANGES and _has_foreign_script_leak(answer, response_language):
        print("Language mixing detected, retrying with stricter instruction...")
        retry_prompt = prompt + f"\n\nIMPORTANT: Your previous attempt mixed languages. Write EVERY sentence in {explicit_lang} script only."
        try:
            retry_answer = _call_llm(retry_prompt, max_tokens=1800)
            retry_answer = _extract_final_answer(retry_answer)
            if retry_answer and not _has_foreign_script_leak(retry_answer, response_language):
                return retry_answer
        except Exception as e:
            print(f"Retry error: {repr(e)}")
        return f"(A clean single-language answer could not be generated by the current model.)\n\n{answer}"

    return answer


def generate_synthesis(
    question: str,
    per_regime_answers: list[dict],
) -> str:

    combined = "\n\n".join(f"[{r['regime']}]\n{r['answer']}" for r in per_regime_answers)

    prompt = f"""You are comparing regulatory guidance across jurisdictions for Ayurveda-related IP and regulatory questions.
Below are answers already generated for each jurisdiction, based strictly on their own regulatory sources.

Question:
{question}

Per-jurisdiction answers:
{combined}

IMPORTANT OUTPUT RULES:
- Return ONLY the final synthesis comparison.
- Do NOT show reasoning or analysis process.
- Do NOT introduce new claims. Only compare the information provided above.
- Highlight key agreements or differences across jurisdictions in 3-5 sentences."""

    try:
        content = _call_llm(prompt, max_tokens=800)
        content = _extract_final_answer(content)
    except Exception as e:
        print(f"Synthesis error: {repr(e)}")
        content = ""

    if not content:
        content = "Synthesis unavailable — please review the per-jurisdiction answers above."

    return content
