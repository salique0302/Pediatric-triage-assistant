"""
app/main.py

FastAPI application for Mumzworld Pediatric Triage Assistant.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime

from app.models import TriageRequest, TriageResponse
from app.triage import run_triage
from app.rag import initialize_rag
from app.products import load_product_catalog
from app.llm import TriageLLMError


app = FastAPI(
    title="Mumzworld Pediatric Triage API",
    description="Pediatric symptom triage assistant for Mumzworld",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("=== Mumzworld Triage API Starting ===")

    try:
        # Initialize RAG (ChromaDB + embeddings)
        initialize_rag()

        # Load product catalog
        load_product_catalog()

        # Prompt templates are loaded lazily via lru_cache

        print("=== Startup Complete ===\n")
    except Exception as e:
        print(f"ERROR during startup: {e}")
        print("Make sure you have run: python scripts/ingest_kb.py")
        raise


@app.get("/")
async def root():
    """Serve the frontend HTML."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if not frontend_path.exists():
        return {"message": "Mumzworld Pediatric Triage API", "status": "running"}
    return FileResponse(frontend_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "kb_loaded": True,
    }


@app.post("/triage", response_model=TriageResponse)
async def triage_endpoint(request: TriageRequest):
    """
    Main triage endpoint.

    Accepts symptom description and child age, returns structured triage response.
    """
    try:
        # Log request (no PII — just metadata)
        print(f"Triage request: age={request.child_age_months}mo, symptoms_length={len(request.symptoms)}")

        # Run triage pipeline
        response = await run_triage(request)

        # Log response metadata
        print(f"Triage response: severity={response.severity}, defer={response.defer_to_doctor}, lang={response.input_language}")

        return response

    except TriageLLMError as e:
        # LLM-specific errors return 503 (service unavailable)
        raise HTTPException(
            status_code=503,
            detail=f"Triage service temporarily unavailable: {str(e)}"
        )
    except Exception as e:
        # Unexpected errors
        print(f"ERROR in triage endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during triage"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
