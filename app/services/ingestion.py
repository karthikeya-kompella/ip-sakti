# app/services/ingestion.py
import uuid
from sqlalchemy.orm import Session

from app.db.pinecone_client import index
from app.services.embedding import embed_texts
from app.models.document import Document


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_document(
    db: Session,
    text: str,
    title: str,
    regime: str,
    language: str,
    doc_type: str,
    source_url: str | None = None,
) -> Document:
    doc = Document(
        title=title,
        regime=regime,
        language=language,
        doc_type=doc_type,
        source_url=source_url,
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]

        vectors = []
        for i in range(len(chunks)):
            metadata = {
                "document_id": doc.id,
                "title": title,
                "regime": regime,
                "language": language,
                "doc_type": doc_type,
                "source_url": source_url or "",
                "chunk_text": chunks[i],
            }
            vectors.append((ids[i], embeddings[i], metadata))

        # Pinecone caps request size at 2MB — upsert in small batches to
        # stay safely under that limit regardless of document size.
        batch_size = 50
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)

        doc.status = "done"
        doc.chunk_count = len(chunks)
        db.commit()
        db.refresh(doc)

    except Exception as e:
        doc.status = "failed"
        db.commit()
        raise e

    return doc
