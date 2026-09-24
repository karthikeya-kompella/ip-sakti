from app.services.generation import client
from app.config import settings

def score_faithfulness(question: str, answer: str, context: str) -> dict:
    prompt = f"""You are evaluating whether an AI-generated answer is faithful to its source context.

Source context:
{context}

Question: {question}

Generated answer:
{answer}

Evaluate ONLY based on whether every claim in the generated answer is actually 
supported by the source context above. Do not evaluate correctness against 
outside knowledge — only check if the answer stays grounded in the given context.

Respond in EXACTLY this format, nothing else:
SCORE: <a number from 0 to 10, where 10 means fully grounded with no unsupported claims>
REASON: <one sentence explaining the score>
"""

    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL_NAME,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""
    return _parse_score(content)


def score_relevancy(question: str, answer: str) -> dict:
    prompt = f"""You are evaluating whether an AI-generated answer actually addresses the question asked.

Question: {question}

Generated answer:
{answer}

Evaluate ONLY whether the answer is relevant to and actually addresses the question. 
Do not evaluate factual correctness.

Respond in EXACTLY this format, nothing else:
SCORE: <a number from 0 to 10, where 10 means fully relevant and directly answers the question>
REASON: <one sentence explaining the score>
"""

    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL_NAME,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or ""
    return _parse_score(content)


def _parse_score(text: str) -> dict:
    score = None
    reason = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except ValueError:
                score = None
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {"score": score, "reason": reason, "raw": text}


def evaluate_rag_response(
    question,
    answer,
    retrieved_chunks,
    reference_answer=None,
):

    context = "\n\n".join(retrieved_chunks)

    faithfulness = score_faithfulness(question, answer, context)
    answer_relevancy = score_relevancy(question, answer)

    result = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }

    if reference_answer:
        result["reference_answer"] = reference_answer

    return result