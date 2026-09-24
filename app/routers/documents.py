# app/routers/documents.py
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.document import Document
from app.services.pdf_extract import extract_text_from_pdf
from app.services.ingestion import ingest_document
from app.db.pinecone_client import index

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

ADMIN_EMAILS = {"karthikeya@gmail.com"}  # your account only


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.post("/upload")
def upload_document(
    title: str = Form(...),
    regime: str = Form(...),
    language: str = Form(...),
    doc_type: str = Form(...),
    source_url: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(str(file_path))
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")

    doc = ingest_document(
        db=db,
        text=text,
        title=title,
        regime=regime,
        language=language,
        doc_type=doc_type,
        source_url=source_url,
    )
    return {
        "id": doc.id,
        "title": doc.title,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
    }


@router.get("")
def list_documents(regime: str | None = None, db: Session = Depends(get_db)):
    query = db.query(Document)
    if regime:
        query = query.filter(Document.regime == regime)
    return query.all()


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    index.delete(filter={"document_id": document_id})

    db.delete(doc)
    db.commit()

    return {"status": "deleted", "document_id": document_id}