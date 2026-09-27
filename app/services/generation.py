# app/services/generation.py
import re
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

SCRIPT_RANGES = {
    "te": r'[\u0C00-\u0C7F]',
    "hi": r'[\u0900-\u097F]',
    "zh": r'[\u4E00-\u9FFF]',
}


def _call_llm(prompt: str, max_tokens: int = 1500) -> str:
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


def _has_foreign_script_leak(text: str, target_code: str) -> bool:
    """Check if the answer contains script from a DIFFERENT language than
    the one requested — this is the actual language-mixing bug."""
    for code, pattern in SCRIPT_RANGES.items():
        if code != target_code and re.search(pattern, text):
            return True
    return False


def generate_answer(
    question: str,
    chunks: list[dict],
    response_language: str | None = None,
) -> str:

    lang_code = response_language if response_language in LANGUAGE_NAMES else "en"
    lang_name = LANGUAGE_NAMES[lang_code]

    if not chunks:
        return f"The available documents do not contain enough information to answer this question. (Requested language: {lang_name})"

    context_parts = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title", "Unknown source")
        chunk_text = c.get("chunk_text", "")
        context_parts.append(f"SOURCE {i}\nTITLE: {title}\n\nCONTENT:\n{chunk_text}\n")

    context = "\n".join(context_parts)

    prompt = f"""You are IP-Sakti Sahayak, a jurisdiction-aware legal and regulatory assistant for Ayurveda IP and drug regulation.

LANGUAGE RULE — THIS IS ABSOLUTE:
Write your ENTIRE answer in {lang_name} ONLY.
Do NOT mix in English, Hindi, Chinese, or any other language.
Do NOT switch languages mid-sentence or mid-paragraph.
Proper nouns, Act names, and section numbers may stay in their original form (e.g. "Section 3(p)", "Drugs and Cosmetics Act"), but every surrounding sentence, explanation, and instruction must be written in {lang_name}.
If you cannot write fluently in {lang_name}, still attempt it — do not fall back to English.

CONTENT RULES:
1. Base your answer ONLY on the retrieved sources below. Do not invent facts, sections, or authorities not present in the sources.
2. Directly answer the user's question first, in one or two clear sentences.
3. Then provide PRACTICAL GUIDANCE based on what the sources say — this means:
   - What the applicable rule, section, or requirement actually says
   - What a person/company would need to DO to comply (e.g. which approval to obtain, which form to file, which authority to approach, what formulation/manufacturing condition applies)
   - Any conditions, exceptions, or limitations mentioned in the source text
4. If the sources describe a process or formulation requirement, lay it out as clear, ordered steps.
5. If the sources are insufficient to answer or to give practical guidance, say so explicitly, in {lang_name}, and do not guess.
6. Do NOT include reasoning traces, meta-commentary, or labels like "Analysis:", "Let me think", or "Here's the answer:". Output ONLY the final answer text itself.

RETRIEVED SOURCES:
{context}

USER QUESTION:
{question}

FINAL ANSWER (entirely in {lang_name}):"""

    try:
        answer = _call_llm(prompt, max_tokens=1500)
        answer = _extract_final_answer(answer)
    except Exception as e:
        print(f"Generation error: {repr(e)}")
        answer = ""

    if not answer:
        return (
            f"(Model unavailable — showing top retrieved source directly.) "
            f"[{chunks[0].get('title','Source')}] {chunks[0].get('chunk_text','')[:500]}"
        )

    if lang_code != "en" and _has_foreign_script_leak(answer, lang_code):
        print(f"Language mixing detected for {lang_name}, retrying with stricter instruction...")
        retry_prompt = prompt + f"\n\nIMPORTANT: Your previous attempt mixed languages. This time, write EVERY sentence in {lang_name} script only. No exceptions."
        try:
            retry_answer = _call_llm(retry_prompt, max_tokens=1500)
            retry_answer = _extract_final_answer(retry_answer)
            if retry_answer and not _has_foreign_script_leak(retry_answer, lang_code):
                return retry_answer
        except Exception as e:
            print(f"Retry error: {repr(e)}")
        return f"(A clean {lang_name}-only answer could not be generated by the current model.)\n\n{answer}"

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
