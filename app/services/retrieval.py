# app/services/retrieval.py
from app.db.pinecone_client import index
from app.services.embedding import embed_texts


def retrieve_chunks(
    query: str,
    regime: str,
    top_k: int = 5
) -> list[dict]:

    print("\n========== RETRIEVAL DEBUG ==========")
    print("QUERY:", query)
    print("REQUESTED REGIME:", regime)
    print("TOP K:", top_k)

    query_embedding = embed_texts([query])[0]
    print("EMBEDDING LENGTH:", len(query_embedding))

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter={"regime": regime},
        include_metadata=True,
    )

    matches = results.get("matches", [])
    print("NUMBER OF RESULTS:", len(matches))

    chunks = []

    for i, match in enumerate(matches):
        meta = match.get("metadata", {})
        score = match.get("score", None)  # Pinecone cosine: higher = more similar

        title = meta.get("title", "Unknown source")
        chunk_regime = meta.get("regime", "UNKNOWN")
        language = meta.get("language", "UNKNOWN")
        doc_type = meta.get("doc_type", "UNKNOWN")
        source_url = meta.get("source_url", None)
        chunk_text_val = meta.get("chunk_text", "")

        print(f"\n--- RESULT {i + 1} ---")
        print("SCORE:", score)
        print("TITLE:", title)
        print("REQUESTED REGIME:", regime)
        print("DOCUMENT REGIME:", chunk_regime)
        print("LANGUAGE:", language)
        print("DOC TYPE:", doc_type)
        print("TEXT:", chunk_text_val[:500])

        if chunk_regime != regime:
            print("WARNING: WRONG REGIME RETURNED!")
            continue

        chunks.append({
            "chunk_text": chunk_text_val,
            "title": title,
            "regime": chunk_regime,
            "language": language,
            "doc_type": doc_type,
            "source_url": source_url,
            "score": score,
        })

    print("\nFINAL VALID CHUNKS:", len(chunks))
    print("=====================================\n")

    return chunks



def compute_confidence(chunks: list[dict]) -> dict:
    """Derive a simple confidence label from the top retrieval score."""
    if not chunks:
        return {"level": "none", "score": 0.0, "note": "No relevant sources found."}

    top_score = chunks[0].get("score") or 0.0

    if top_score >= 0.75:
        level = "high"
        note = "Strong source match."
    elif top_score >= 0.55:
        level = "medium"
        note = "Moderate source match — verify against the cited passage."
    else:
        level = "low"
        note = "Weak source match — treat this answer with caution and consult a human IP facilitator."

    return {"level": level, "score": round(top_score, 3), "note": note}
