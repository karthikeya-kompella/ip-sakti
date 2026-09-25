
# app/main.py
from app import models  # ensures all models are registered with SQLAlchemy

from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, regimes, documents, query



from fastapi import FastAPI
from app.db.database import Base, engine
from app.models import user, conversation, message, document

try:
    Base.metadata.create_all(bind=engine)
    print("Database tables ready.")
except Exception as e:
    print(f"WARNING: Could not create tables at startup: {repr(e)}")

app = FastAPI(title="IP-SAKTI Sahayak API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(regimes.router)
app.include_router(documents.router)
app.include_router(query.router)
@app.get("/")
def root():
    return {"status": "IP-SAKTI Sahayak API is running"}
>>>>>>> dc7b48e (initialize database tables)
