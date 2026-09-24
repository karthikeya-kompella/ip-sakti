from app.services.generation import client
from app.config import settings
from app.services.retrieval import retrieve_chunks


# ---------------------------------------------------------
# 1. Evaluate whether retrieved chunks are useful
# ---------------------------------------------------------

def evaluate_retrieval(
    question: str,
    chunks: list[dict]
) -> dict:

    if not chunks:
        return {
            "score": 0.0,
            "relevant_chunks": [],
            "reason": "No chunks were retrieved."
        }

    results = []

    for chunk in chunks:

        chunk_text = chunk.get("chunk_text", "")

        prompt = f"""
You are a retrieval evaluator for a legal RAG system.

User question:
{question}

Retrieved document:
{chunk_text}

Evaluate how relevant this document is for answering the user's question.

Consider:
- Does it contain information needed to answer the question?
- Is it about the correct legal/regulatory topic?
- Does it provide useful evidence?
- Does it belong to the relevant jurisdiction/topic?

Give a score from 0 to 10.

Respond EXACTLY:

SCORE: <number>
REASON: <one short sentence>
"""

        response = client.chat.completions.create(
            model=settings.GENERATION_MODEL_NAME,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        content = response.choices[0].message.content or ""

        score = parse_score(content)

        results.append({
            "chunk": chunk,
            "score": score,
            "reason": content
        })

    # Average retrieval quality
    valid_scores = [
        r["score"]
        for r in results
        if r["score"] is not None
    ]

    average_score = (
        sum(valid_scores) / len(valid_scores)
        if valid_scores
        else 0.0
    )

    # Keep only relevant chunks
    relevant_chunks = [
        r["chunk"]
        for r in results
        if r["score"] is not None and r["score"] >= 6
    ]

    return {
        "score": round(average_score, 2),
        "relevant_chunks": relevant_chunks,
        "details": results,
    }


# ---------------------------------------------------------
# 2. Generate a better query for corrective retrieval
# ---------------------------------------------------------

def generate_corrective_query(
    question: str,
    regime: str
) -> str:

    prompt = f"""
You are a query reformulation system for a legal RAG application.

Original question:
{question}

Jurisdiction:
{regime}

The first retrieval attempt did not retrieve sufficiently
relevant information.

Rewrite the question into a better search query.

The query should:
- preserve the original meaning
- include important legal terms
- include relevant section names/numbers if appropriate
- focus on the specified jurisdiction
- be suitable for document retrieval

Return ONLY the improved search query.
"""

    response = client.chat.completions.create(
        model=settings.GENERATION_MODEL_NAME,
        max_tokens=120,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return (
        response.choices[0].message.content or question
    ).strip()


# ---------------------------------------------------------
# 3. Corrective retrieval
# ---------------------------------------------------------

def corrective_retrieval(
    question: str,
    regime: str,
    original_chunks: list[dict],
    top_k: int = 5
) -> tuple[list[dict], dict]:

    improved_query = generate_corrective_query(
        question,
        regime
    )

    new_chunks = retrieve_chunks(
        query=improved_query,
        regime=regime,
        top_k=top_k * 2
    )

    # -----------------------------------------------------
    # Merge original + corrected chunks
    # -----------------------------------------------------

    combined = []

    seen = set()

    for chunk in original_chunks + new_chunks:

        text = chunk.get("chunk_text", "").strip()

        if not text:
            continue

        # Avoid duplicate chunks
        chunk_id = (
            chunk.get("id")
            or chunk.get("chunk_id")
            or text[:200]
        )

        if chunk_id not in seen:
            seen.add(chunk_id)
            combined.append(chunk)

    # -----------------------------------------------------
    # Re-evaluate combined context
    # -----------------------------------------------------

    evaluation = evaluate_retrieval(
        question,
        combined
    )

    final_chunks = evaluation["relevant_chunks"]

    # If evaluator rejected everything,
    # fall back to the best retrieved chunks.
    if not final_chunks:
        final_chunks = combined[:top_k]

    return final_chunks[:top_k], {
        "corrective_query": improved_query,
        "retrieval_score_after_correction": evaluation["score"],
        "corrected": True,
    }


# ---------------------------------------------------------
# 4. Main CRAG pipeline
# ---------------------------------------------------------

def crag_retrieve(
    question: str,
    regime: str,
    top_k: int = 5,
    threshold: float = 6.0
) -> tuple[list[dict], dict]:

    # FIRST RETRIEVAL
    initial_chunks = retrieve_chunks(
        query=question,
        regime=regime,
        top_k=top_k
    )

    # EVALUATE RETRIEVAL
    initial_evaluation = evaluate_retrieval(
        question,
        initial_chunks
    )

    initial_score = initial_evaluation["score"]

    # -----------------------------------------------------
    # GOOD RETRIEVAL
    # -----------------------------------------------------

    if initial_score >= threshold:

        final_chunks = initial_evaluation["relevant_chunks"]

        if not final_chunks:
            final_chunks = initial_chunks

        return final_chunks[:top_k], {
            "crag_triggered": False,
            "initial_retrieval_score": initial_score,
            "final_retrieval_score": initial_score,
            "reason": "Initial retrieval was sufficiently relevant."
        }

    # -----------------------------------------------------
    # BAD RETRIEVAL → CORRECTIVE RETRIEVAL
    # -----------------------------------------------------

    final_chunks, correction_info = corrective_retrieval(
        question=question,
        regime=regime,
        original_chunks=initial_chunks,
        top_k=top_k
    )

    return final_chunks, {
        "crag_triggered": True,
        "initial_retrieval_score": initial_score,
        **correction_info,
        "reason": "Initial retrieval was below the quality threshold."
    }


# ---------------------------------------------------------
# Score parser
# ---------------------------------------------------------

def parse_score(text: str):

    for line in text.strip().splitlines():

        line = line.strip()

        if line.upper().startswith("SCORE:"):

            try:
                score = float(
                    line.split(":", 1)[1].strip()
                )

                return max(0.0, min(10.0, score))

            except ValueError:
                return None

    return None