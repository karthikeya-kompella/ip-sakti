from openai import OpenAI
from app.config import settings

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "zh": "Chinese",
}
import re

def _contains_script(text: str, language_code: str) -> bool:
    """Check whether text actually contains characters from the target script."""
    script_ranges = {
        "te": r'[\u0C00-\u0C7F]',   # Telugu unicode block
        "hi": r'[\u0900-\u097F]',   # Devanagari (Hindi) unicode block
        "zh": r'[\u4E00-\u9FFF]',   # CJK unicode block
    }
    pattern = script_ranges.get(language_code)
    if not pattern:
        return True  # no script check defined (e.g. English) — assume fine
    return bool(re.search(pattern, text))


def _translate_text(text: str, target_language_name: str, target_language_code: str) -> str:
    prompt = f"""Translate the following text into {target_language_name}.
You MUST write the ENTIRE output in {target_language_name} script — do not use English or Latin script.
Preserve all facts, section numbers, and legal/regulatory terms exactly.
Output ONLY the translated text, nothing else.

Text to translate:
{text}"""

    try:
        translated = _call_llm(prompt, max_tokens=1500)
        translated = _extract_final_answer(translated)

        if translated and _contains_script(translated, target_language_code):
            return translated

        # First attempt failed the script check — retry once, more forcefully
        print(f"Translation to {target_language_name} did not produce {target_language_name} script, retrying...")
        retry_prompt = f"""Translate this into {target_language_name} language, using {target_language_name} script ONLY.
This is critical: your entire response must be written in {target_language_name} characters, not English.

{text}"""
        retry = _call_llm(retry_prompt, max_tokens=1500)
        retry = _extract_final_answer(retry)

        if retry and _contains_script(retry, target_language_code):
            return retry

        # Still failed — be honest about it rather than silently returning English
        return f"(Translation to {target_language_name} was not available — showing English answer instead)\n\n{text}"

    except Exception as e:
        print(f"Translation error: {repr(e)}")
        return text

def _call_llm(prompt: str, max_tokens: int = 1200) -> str:
    print(f"Calling LLM ({settings.GENERATION_MODEL_NAME})...")
    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL_NAME,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _extract_final_answer(text: str) -> str:
    """If the model leaked reasoning before its real answer, take only the
    last paragraph block, which is consistently where the actual answer lands."""
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


def _translate_text(text: str, target_language_name: str, target_language_code: str) -> str:
    """Dedicated translation pass, separate from RAG generation."""
    prompt = f"""Translate the following text into {target_language_name}.
Preserve all facts, section numbers, and legal/regulatory terms exactly.
Output ONLY the translated text. No preamble, no explanation.

Text to translate:
{text}"""

    try:
        translated = _call_llm(prompt, max_tokens=1500)
        translated = _extract_final_answer(translated)
        return translated if translated else text
    except Exception as e:
        print(f"Translation error: {repr(e)}")
        return text


def generate_answer(
    question: str,
    chunks: list[dict],
    response_language: str | None = None,
) -> str:

    # Determine the requested response language
    lang_name = (
        LANGUAGE_NAMES.get(response_language)
        if response_language
        else "the same language as the user's question"
    )

    if not chunks:
        base_message = (
            "The available documents do not contain enough information "
            "to answer this question."
        )

        if response_language and lang_name != "English":
            return _translate_text(base_message, lang_name)

        return base_message

    # ---------------------------------------------------------
    # BUILD RETRIEVED CONTEXT
    # ---------------------------------------------------------

    context_parts = []

    for i, c in enumerate(chunks, start=1):
        title = c.get("title", "Unknown source")
        chunk_text = c.get("chunk_text", "")

        context_parts.append(
            f"SOURCE {i}\n"
            f"TITLE: {title}\n\n"
            f"CONTENT:\n{chunk_text}\n"
        )

    context = "\n".join(context_parts)

    # ---------------------------------------------------------
    # GENERATION PROMPT
    # ---------------------------------------------------------

    prompt = f"""
You are the final-answer generator for IP-Sakti Sahayak,
a jurisdiction-aware legal and regulatory information assistant.

Answer the user's question using ONLY the retrieved sources below.

LANGUAGE RULES:

1. Answer in the SAME LANGUAGE used by the user.
2. The required response language is: {lang_name}
3. Do NOT automatically answer in English.
4. Do NOT translate the user's question into another language.
5. Maintain the same language throughout the answer.
6. If the user mixes languages, answer primarily in the language
   used for the main question.
7. Use natural and grammatically correct language.
8. Do not unnecessarily mix English sentences into a non-English answer.
9. Official legal and regulatory terms may remain in their original
   form when necessary.

LEGAL TERMINOLOGY RULES:

- Preserve official names of laws and authorities.
- Preserve section numbers exactly.
- Preserve CFR references exactly.
- Preserve article/rule numbers exactly.
- Preserve official abbreviations such as FDA, NDA, OTC,
  FD&C Act, etc.
- Do not invent or modify legal references.

STRICT RULES:

1. Answer the user's question directly and concisely.
2. Use ONLY information supported by the retrieved sources.
3. Do NOT use outside knowledge.
4. Do NOT invent legal sections, rules, dates, authorities,
   requirements, or procedures.
5. If the retrieved sources are insufficient, explicitly say:

   "The available sources do not contain enough information
   to answer this question."

   Express this statement in the user's language.
6. Do NOT include reasoning steps.
7. Do NOT include meta commentary.
8. Do NOT include labels such as "Analysis", "Reasoning",
   or "Thought process".
9. Distinguish clearly between jurisdictions.
10. Do not make unsupported legal conclusions.

RETRIEVED SOURCES:

{context}

USER QUESTION:

{question}

FINAL INSTRUCTION:

Provide ONLY the final answer to the user's question.
Answer in the SAME LANGUAGE as the user's question.
"""

    # ---------------------------------------------------------
    # CALL LLM
    # ---------------------------------------------------------

    try:
        answer = _call_llm(prompt, max_tokens=1200)
        answer = _extract_final_answer(answer)

    except Exception as e:
        print(f"Generation error: {repr(e)}")
        answer = ""

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------

    if not answer:
        answer = (
            f"(Model unavailable — showing top retrieved source directly.) "
            f"[{chunks[0].get('title', 'Source')}] "
            f"{chunks[0].get('chunk_text', '')[:500]}"
        )

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
- Do NOT show reasoning or analysis.
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