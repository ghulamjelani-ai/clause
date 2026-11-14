from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Any
from pydantic import BaseModel, StringConstraints
from app.services.chat import RAGService, Answer
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAGService and store it in app state
    app.state.rag_service = RAGService()
    try:
        yield
    finally:
        # Clean up resources
        app.state.rag_service.close()

app = FastAPI(title="Clause API", lifespan=lifespan)

# --- CORS Configuration ---
# This allows your Node.js frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend URL (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health() -> str:
    """Return a simple health check response."""
    return "Clause API is healthy"

class AskQuestion(BaseModel):
    question: Annotated[
        str, StringConstraints(min_length=1, max_length=1000, strip_whitespace=True)
    ]

@app.post("/ask", response_model=Answer)
def ask_query(body: AskQuestion) -> Any:
    return app.state.rag_service.ask(query=body.question)
    