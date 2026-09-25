from app import models  # ensures all models are registered with SQLAlchemy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, regimes, documents, query

from app.db.database import Base, engine
from app.models import user, conversation, message, document


# ==========================================
# CREATE DATABASE TABLES
# ==========================================

try:
    Base.metadata.create_all(bind=engine)
    print("Database tables ready.")
except Exception as e:
    print(f"WARNING: Could not create tables at startup: {repr(e)}")


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="IP-SAKTI Sahayak API"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ==========================================
# ROUTERS
# ==========================================

app.include_router(auth.router)
app.include_router(regimes.router)
app.include_router(documents.router)
app.include_router(query.router)


# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():
    return {
        "status": "IP-SAKTI Sahayak API is running"
    }
