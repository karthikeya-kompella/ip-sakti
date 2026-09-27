# app/services/generation.py

import re
from openai import OpenAI

from app.config import settings


# ============================================================
# OpenRouter / OpenAI Client
# ============================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY,
)


# ============================================================
# Supported Languages
# ============================================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "zh": "Chinese",
}


# ============================================================
# Script Detection
# ============================================================

SCRIPT_RANGES = {
    "te": r"[\u0C00-\u0C7F]",      # Telugu
    "hi": r"[\u0900-\u097F]",      # Devanagari / Hindi
    "zh": r"[\u4E00-\u9FFF]",      # Chinese
}


# ============================================================
# LLM Call
# ============================================================

def _call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """
    Send a prompt to the configured OpenRouter model
    and return the generated response.
    """

    print(
        f"Calling LLM ({settings.GENERATION_MODEL_NAME})..."
    )

    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL_NAME,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    content = response.choices[0].message.content or ""

    print(f"RAW ANSWER LENGTH: {len(content)}")
    print(f"RAW ANSWER PREVIEW: {content[:300]}")

    return content.strip()


# ============================================================
# Answer Extraction
# ============================================================

def _extract_final_answer(text: str) -> str:
    """
    Clean the model output.

    We intentionally do NOT select the longest paragraph because
    doing so can accidentally remove important parts of the answer.

    The prompt already instructs the model to return only the final
    answer, so we preserve the complete response.
    """

    if not text:
        return ""

    text = text.strip()

    # Handle occasional reasoning leakage where the model
    # explicitly writes "Final Answer:"
    markers = [
        "final answer:",
        "final answer",
    ]

    lower = text.lower()

    latest_index = -1
    selected_marker = None

    for marker in markers:
        index = lower.rfind(marker)

        if index > latest_index:
            latest_index = index
            selected_marker = marker

    if latest_index != -1 and selected_marker:
        candidate = text[
            latest_index + len(selected_marker):
        ].strip(" :\n")

        # Only use the extracted section if it contains
        # meaningful content.
        if len(candidate) > 20:
            return candidate

    return text


# ============================================================
# Foreign Script Detection
# ============================================================

def _has_foreign_script_leak(
    text: str,
    target_code: str,
) -> bool:
    """
    Detect whether the generated answer contains a script
    belonging to another supported language.

    Example:

    Target = Telugu

    Telugu text + Hindi text
        -> True

    Telugu text + English legal terms
        -> False

    English legal terms are intentionally not checked here
    because legal/regulatory names such as:

        Patents Act
        Section 3(p)
        Form 1
        AYUSH

    are allowed to remain in their original form.
    """

    if not text:
        return False

    for code, pattern in SCRIPT_RANGES.items():

        if code == target_code:
            continue

        if re.search(pattern, text):
            return True

    return False


# ============================================================
# Generate Answer
# ============================================================

def generate_answer(
    question: str,
    chunks: list[dict],
    response_language: str | None = None,
) -> str:

    # --------------------------------------------------------
    # Determine requested language
    # --------------------------------------------------------

    lang_code = (
        response_language
        if response_language in LANGUAGE_NAMES
        else "en"
    )

    lang_name = LANGUAGE_NAMES[lang_code]

    # --------------------------------------------------------
    # No retrieved documents
    # --------------------------------------------------------

    if not chunks:
        return (
            "The available documents do not contain enough "
            f"information to answer this question. "
            f"(Requested language: {lang_name})"
        )

    # --------------------------------------------------------
    # Build retrieved context
    # --------------------------------------------------------

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):

        title = chunk.get(
            "title",
            "Unknown source",
        )

        chunk_text = chunk.get(
            "chunk_text",
            "",
        )

        context_parts.append(
            f"SOURCE {i}\n"
            f"TITLE: {title}\n\n"
            f"CONTENT:\n"
            f"{chunk_text}\n"
        )

    context = "\n".join(context_parts)

    # --------------------------------------------------------
    # Main generation prompt
    # --------------------------------------------------------

    prompt = f"""
You are IP-Sakti Sahayak, a jurisdiction-aware legal and
regulatory assistant for Ayurveda intellectual property and
drug regulation.

============================================================
LANGUAGE RULE — ABSOLUTE
============================================================

Write your ENTIRE answer in {lang_name} ONLY.

Do NOT mix in English, Hindi, Chinese, Telugu, or any other
language unnecessarily.

Do NOT switch languages in the middle of a sentence or
paragraph.

Proper nouns, official authority names, Act names, legal
document names, technical terms, form numbers, section
numbers, and citations may remain in their original form.

Examples:

- Section 3(p)
- Patents Act, 1970
- Drugs and Cosmetics Act
- AYUSH
- Form 1

However, all surrounding explanation, instructions,
reasoning, and guidance must be written in {lang_name}.

If you cannot write perfectly in {lang_name}, still attempt
to answer in {lang_name}. Do NOT automatically fall back
to English.

============================================================
CONTENT RULES
============================================================

1. Base your answer ONLY on the retrieved sources below.

2. Do NOT invent:
   - laws
   - sections
   - rules
   - authorities
   - dates
   - forms
   - requirements
   - procedures
   - penalties
   - regulatory claims

3. Directly answer the user's question first.

4. After the direct answer, provide practical guidance based
   only on what the retrieved sources support.

5. Where applicable, explain:
   - the applicable rule or section
   - what the rule requires
   - what the applicant/company needs to do
   - which authority is involved
   - which form or document is mentioned
   - relevant conditions
   - relevant exceptions
   - relevant limitations

6. If the sources describe a process, present the process
   as clear ordered steps.

7. If the retrieved sources are insufficient to answer the
   question, explicitly say that the available sources are
   insufficient.

8. Do NOT guess or fill missing information from general
   knowledge.

9. Do NOT provide legal advice as if you are a lawyer.
   Present the retrieved regulatory information accurately
   and indicate when professional/legal confirmation may
   be necessary.

10. Do NOT include:
    - reasoning traces
    - chain-of-thought
    - internal analysis
    - meta-commentary
    - "Analysis:"
    - "Let me think"
    - "Here's my reasoning"
    - "Final answer:"
    - similar internal-process labels

11. Output ONLY the final answer.

============================================================
RETRIEVED SOURCES
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
FINAL ANSWER
============================================================

Write the complete answer entirely in {lang_name}.
"""

    # --------------------------------------------------------
    # Call model
    # --------------------------------------------------------

    try:

        answer = _call_llm(
            prompt,
            max_tokens=1500,
        )

        print(
            f"ANSWER BEFORE EXTRACTION LENGTH: "
            f"{len(answer)}"
        )

        answer = _extract_final_answer(answer)

        print(
            f"ANSWER AFTER EXTRACTION LENGTH: "
            f"{len(answer)}"
        )

    except Exception as e:

        print(
            f"Generation error: {repr(e)}"
        )

        answer = ""

    # --------------------------------------------------------
    # Model returned nothing
    # --------------------------------------------------------

    if not answer:

        first_source_title = chunks[0].get(
            "title",
            "Source",
        )

        first_source_text = chunks[0].get(
            "chunk_text",
            "",
        )

        return (
            "(Model unavailable — showing top retrieved "
            "source directly.) "
            f"[{first_source_title}] "
            f"{first_source_text[:500]}"
        )

    # --------------------------------------------------------
    # Language mixing detection
    # --------------------------------------------------------

    if (
        lang_code != "en"
        and _has_foreign_script_leak(
            answer,
            lang_code,
        )
    ):

        print(
            f"Language mixing detected for {lang_name}. "
            "Retrying with stricter instruction..."
        )

        retry_prompt = f"""
IMPORTANT LANGUAGE CORRECTION.

Your previous response contained text written in a script
belonging to another supported language.

Generate the answer again.

TARGET LANGUAGE:
{lang_name}

STRICT REQUIREMENT:

Write the explanation entirely in {lang_name}.

Do NOT use Hindi, Telugu, Chinese, or another non-target
language in the explanation.

Official legal names, Act names, section numbers, form
numbers, authority names, and technical/legal terminology
may remain in their original form when necessary.

Do NOT include reasoning.

Do NOT explain that you are retrying.

Return ONLY the final answer.

ORIGINAL INSTRUCTIONS:
{prompt}
"""

        try:

            retry_answer = _call_llm(
                retry_prompt,
                max_tokens=1500,
            )

            retry_answer = _extract_final_answer(
                retry_answer
            )

            if (
                retry_answer
                and not _has_foreign_script_leak(
                    retry_answer,
                    lang_code,
                )
            ):
                print(
                    f"Clean {lang_name} response generated."
                )

                return retry_answer

        except Exception as e:

            print(
                f"Retry error: {repr(e)}"
            )

        # ----------------------------------------------------
        # If retry also fails
        # ----------------------------------------------------

        return (
            f"(A clean {lang_name}-only answer could not "
            f"be generated by the current model.)\n\n"
            f"{answer}"
        )

    # --------------------------------------------------------
    # Normal successful response
    # --------------------------------------------------------

    return answer


# ============================================================
# Cross-Jurisdiction Synthesis
# ============================================================

def generate_synthesis(
    question: str,
    per_regime_answers: list[dict],
) -> str:
    """
    Generate a concise comparison across jurisdictions.

    Each jurisdiction answer is assumed to already be based
    on its own retrieved regulatory sources.
    """

    if not per_regime_answers:

        return (
            "Synthesis unavailable — no jurisdiction answers "
            "were provided."
        )

    # --------------------------------------------------------
    # Combine jurisdiction answers
    # --------------------------------------------------------

    combined_parts = []

    for regime in per_regime_answers:

        regime_name = regime.get(
            "regime",
            "Unknown jurisdiction",
        )

        regime_answer = regime.get(
            "answer",
            "",
        )

        combined_parts.append(
            f"[{regime_name}]\n"
            f"{regime_answer}"
        )

    combined = "\n\n".join(
        combined_parts
    )

    # --------------------------------------------------------
    # Synthesis prompt
    # --------------------------------------------------------

    prompt = f"""
You are comparing regulatory guidance across jurisdictions
for Ayurveda-related intellectual property and regulatory
questions.

The jurisdiction-specific answers below were generated from
their respective regulatory sources.

============================================================
QUESTION
============================================================

{question}

============================================================
PER-JURISDICTION ANSWERS
============================================================

{combined}

============================================================
OUTPUT RULES
============================================================

1. Return ONLY the final synthesis.

2. Do NOT show reasoning or analysis.

3. Do NOT introduce new legal or regulatory claims.

4. Do NOT invent information that is absent from the
   jurisdiction-specific answers.

5. Compare ONLY the information provided.

6. Clearly identify important similarities and differences.

7. Keep the synthesis concise.

8. Use approximately 3–5 sentences.

9. If the jurisdiction-specific answers contain insufficient
   information for a particular comparison, explicitly state
   that the available information is insufficient.

10. Do NOT rank jurisdictions.

11. Do NOT state that one jurisdiction is "better" or "worse".

12. Do NOT provide a legal conclusion beyond the information
    contained in the supplied answers.

13. Output ONLY the final synthesis.
"""

    # --------------------------------------------------------
    # Call model
    # --------------------------------------------------------

    try:

        content = _call_llm(
            prompt,
            max_tokens=800,
        )

        content = _extract_final_answer(
            content
        )

    except Exception as e:

        print(
            f"Synthesis error: {repr(e)}"
        )

        content = ""

    # --------------------------------------------------------
    # Empty response fallback
    # --------------------------------------------------------

    if not content:

        content = (
            "Synthesis unavailable — please review the "
            "per-jurisdiction answers above."
        )

    return content
