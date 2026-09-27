from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_current_user
from app.models.user import User

from app.models.schemas.query import (
    QueryRequest,
    QueryResponse,
    CompareQueryRequest,
    CompareQueryResponse,
)

from app.services.retrieval import (
    retrieve_chunks,
    compute_confidence,
)

from app.services.generation import (
    generate_answer,
    generate_synthesis,
    check_ambiguity,
)

from app.services.cache import (
    query_cache,
    make_cache_key,
)


router = APIRouter(
    prefix="/api/v1/query",
    tags=["query"],
)


# ============================================================
# SINGLE JURISDICTION QUERY
# ============================================================

@router.post(
    "",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
):

    print("\n================ QUERY ================")

    print(
        "QUESTION:",
        request.question
    )

    print(
        "REGIME:",
        request.regime
    )

    print(
        "LANGUAGE:",
        request.response_language
    )

    print(
        "TOP K:",
        request.top_k
    )


    # ========================================================
    # AMBIGUITY CHECK
    # ========================================================

    try:

        ambiguity_check = check_ambiguity(
            request.question
        )

        print(
            "AMBIGUITY CHECK:",
            ambiguity_check
        )

    except Exception as e:

        print(
            "❌ AMBIGUITY CHECK ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Ambiguity check failed.",
        )


    # --------------------------------------------------------
    # QUESTION IS NOT CLEAR
    # --------------------------------------------------------

    if not ambiguity_check.get(
        "is_clear",
        True,
    ):

        print(
            "⚠️ QUESTION REQUIRES CLARIFICATION"
        )

        return QueryResponse(
            answer="",
            regime=request.regime,
            citations=[],
            confidence={
                "level": "none",
                "score": 0.0,
                "note": "Question requires clarification.",
            },
            needs_clarification=True,
            clarifying_questions=ambiguity_check.get(
                "clarifying_questions",
                [],
            ),
        )


    print(
        "✅ QUESTION IS CLEAR"
    )


    # ========================================================
    # CACHE
    # ========================================================

    USE_CACHE = False

    cache_key = make_cache_key(
        request.question,
        request.regime,
        request.top_k,
    )

    if USE_CACHE:

        cached = query_cache.get(
            cache_key
        )

        if cached:

            print(
                "⚠️ RETURNING CACHED RESULT"
            )

            print(
                "========================================\n"
            )

            return cached


    # ========================================================
    # RETRIEVAL
    # ========================================================

    try:

        chunks = retrieve_chunks(
            query=request.question,
            regime=request.regime,
            top_k=request.top_k,
        )

    except Exception as e:

        print(
            "❌ RETRIEVAL ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Retrieval failed.",
        )


    print(
        "\nRETRIEVED CHUNKS:",
        len(chunks)
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    try:

        confidence = compute_confidence(
            chunks
        )

    except Exception as e:

        print(
            "❌ CONFIDENCE ERROR:",
            repr(e)
        )

        confidence = {
            "level": "none",
            "score": 0.0,
            "note": "Confidence could not be calculated.",
        }


    print(
        "CONFIDENCE:",
        confidence
    )


    # ========================================================
    # DEBUG RETRIEVED CHUNKS
    # ========================================================

    for i, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"\n--- CHUNK {i} ---"
        )

        print(
            "TITLE:",
            chunk.get(
                "title",
                "N/A",
            )
        )

        print(
            "REGIME:",
            chunk.get(
                "regime",
                "N/A",
            )
        )

        print(
            "LANGUAGE:",
            chunk.get(
                "language",
                "N/A",
            )
        )

        print(
            "SCORE:",
            chunk.get(
                "score",
                "N/A",
            )
        )

        print(
            "TEXT:",
            chunk.get(
                "chunk_text",
                "",
            )[:500]
        )


    # ========================================================
    # NO RETRIEVED CONTEXT
    # ========================================================

    if not chunks:

        answer = (
            "The available documents do not contain enough "
            "information to answer this question."
        )

        result = QueryResponse(
            answer=answer,
            regime=request.regime,
            citations=[],
            confidence=confidence,
            needs_clarification=False,
            clarifying_questions=[],
        )

        return result


    # ========================================================
    # GENERATION
    # ========================================================

    try:

        answer = generate_answer(
            request.question,
            chunks,
            request.response_language,
        )

    except Exception as e:

        print(
            "❌ GENERATION ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Answer generation failed.",
        )


    print(
        "\nGENERATED ANSWER:"
    )

    print(answer)


    # ========================================================
    # RESPONSE
    # ========================================================

    result = QueryResponse(
        answer=answer,
        regime=request.regime,
        citations=chunks,
        confidence=confidence,
        needs_clarification=False,
        clarifying_questions=[],
    )


    # ========================================================
    # CACHE
    # ========================================================

    if USE_CACHE:

        query_cache[cache_key] = result


    print(
        "========================================\n"
    )

    return result


# ============================================================
# COMPARE JURISDICTIONS
# ============================================================

@router.post(
    "/compare",
    response_model=CompareQueryResponse,
)
def compare_query(
    request: CompareQueryRequest,
    current_user: User = Depends(get_current_user),
):

    print(
        "\n============== COMPARE QUERY =============="
    )

    print(
        "QUESTION:",
        request.question
    )

    print(
        "REGIMES:",
        request.regimes
    )

    print(
        "TOP K:",
        request.top_k
    )


    # ========================================================
    # AMBIGUITY CHECK
    # ========================================================

    try:

        ambiguity_check = check_ambiguity(
            request.question
        )

        print(
            "AMBIGUITY CHECK:",
            ambiguity_check
        )

    except Exception as e:

        print(
            "❌ AMBIGUITY CHECK ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail="Ambiguity check failed.",
        )


    # --------------------------------------------------------
    # STOP IF QUESTION IS UNCLEAR
    # --------------------------------------------------------

    if not ambiguity_check.get(
        "is_clear",
        True,
    ):

        print(
            "⚠️ COMPARE QUESTION REQUIRES CLARIFICATION"
        )

        # Depending on your CompareQueryResponse schema,
        # this may need additional fields.
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Question requires clarification.",
                "clarifying_questions":
                    ambiguity_check.get(
                        "clarifying_questions",
                        [],
                    ),
            },
        )


    # ========================================================
    # PER-REGIME ANSWERS
    # ========================================================

    per_regime_answers = []


    # ========================================================
    # PROCESS EACH JURISDICTION
    # ========================================================

    for regime in request.regimes:

        print(
            f"\n========== REGIME: {regime} =========="
        )


        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        try:

            chunks = retrieve_chunks(
                query=request.question,
                regime=regime,
                top_k=request.top_k,
            )

        except Exception as e:

            print(
                f"❌ RETRIEVAL ERROR for {regime}:",
                repr(e)
            )

            chunks = []


        print(
            f"Retrieved {len(chunks)} chunks for {regime}"
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        try:

            confidence = compute_confidence(
                chunks
            )

        except Exception as e:

            print(
                f"❌ CONFIDENCE ERROR for {regime}:",
                repr(e)
            )

            confidence = {
                "level": "none",
                "score": 0.0,
                "note": (
                    "Confidence could not be calculated."
                ),
            }


        print(
            f"Confidence for {regime}:",
            confidence
        )


        # ----------------------------------------------------
        # DEBUG CHUNKS
        # ----------------------------------------------------

        for i, chunk in enumerate(
            chunks,
            start=1,
        ):

            print(
                f"\n--- {regime} CHUNK {i} ---"
            )

            print(
                "TITLE:",
                chunk.get(
                    "title",
                    "N/A",
                )
            )

            print(
                "REGIME:",
                chunk.get(
                    "regime",
                    "N/A",
                )
            )

            print(
                "LANGUAGE:",
                chunk.get(
                    "language",
                    "N/A",
                )
            )

            print(
                "SCORE:",
                chunk.get(
                    "score",
                    "N/A",
                )
            )

            print(
                "TEXT:",
                chunk.get(
                    "chunk_text",
                    "",
                )[:300]
            )


        # ====================================================
        # GENERATION
        # ====================================================

        if chunks:

            try:

                answer = generate_answer(
                    request.question,
                    chunks,
                )

            except Exception as e:

                print(
                    f"❌ GENERATION ERROR for {regime}:",
                    repr(e)
                )

                answer = (
                    "Answer generation failed for this "
                    "jurisdiction."
                )

        else:

            answer = (
                "The available documents do not contain "
                "enough information to answer this question."
            )


        # ====================================================
        # STORE RESULT
        # ====================================================

        per_regime_answers.append(
            {
                "regime": regime,
                "answer": answer,
                "citations": chunks,
                "confidence": confidence,
            }
        )


    # ========================================================
    # CROSS-JURISDICTION SYNTHESIS
    # ========================================================

    try:

        synthesis = generate_synthesis(
            request.question,
            per_regime_answers,
        )

    except Exception as e:

        print(
            "❌ SYNTHESIS ERROR:",
            repr(e)
        )

        synthesis = (
            "Synthesis could not be generated. "
            "Please review the individual jurisdiction answers."
        )


    print(
        "============================================\n"
    )


    return CompareQueryResponse(
        per_regime_answers=per_regime_answers,
        synthesis=synthesis,
    )
