"""
FastAPI application entrypoint exposing the /ingest endpoint for Phase 1.
- Accepts multipart file upload (CSV or JSON) via python-multipart
- Delegates ingestion to IngestionService (pandas parsing, normalization, SQLite store, FAISS index)
- Provides simple /health endpoint
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
import time
import sqlite3

from app.services.ingestion import IngestionService
from app.services.search import SearchService
from app.services.tracking import TrackingService
from app.services.learning import LearningService

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")

# Paths (configurable via environment variables)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.getenv("DB_PATH") or os.path.join(BASE_DIR, "db", "products.db")
INDEX_PATH = os.getenv("FAISS_INDEX_PATH") or os.path.join(BASE_DIR, "indexes", "faiss.index")

# Ensure folders exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

app = FastAPI(title="Contextual Search - Phase 1 Ingestion")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service (lazy-loads model and index on first use)
ingestion_service = IngestionService(db_path=DB_PATH, index_path=INDEX_PATH)
search_service = SearchService(db_path=DB_PATH, index_path=INDEX_PATH)
tracking_service = TrackingService(db_path=DB_PATH)
learning_service = LearningService(db_path=DB_PATH)


@app.get("/health")
def health():
    """Basic health check."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Ingest a dataset uploaded as multipart file (CSV or JSON).
    - Parses with pandas
    - Normalizes fields
    - Embeds searchable text
    - Stores products in SQLite and vectors in FAISS
    Returns ingestion stats with latency metrics.
    """
    start_time = time.perf_counter()
    try:
        filename = file.filename or "uploaded"
        content_type = file.content_type or "application/octet-stream"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file upload")

        logger.info("POST /ingest start: filename=%s, size=%d bytes", filename, len(content))
        result = ingestion_service.ingest_bytes(
            content_bytes=content,
            filename=filename,
            content_type=content_type,
        )
        elapsed = time.perf_counter() - start_time
        result["latency_ms"] = round(elapsed * 1000, 2)
        logger.info(
            "POST /ingest success: inserted=%d, skipped=%d, latency=%.2fms",
            result.get("inserted", 0),
            result.get("skipped", 0),
            result["latency_ms"],
        )
        return JSONResponse(result)
    except HTTPException as he:
        elapsed = time.perf_counter() - start_time
        logger.warning("POST /ingest failed with status %d: latency=%.2fms", he.status_code, elapsed * 1000)
        raise he
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.exception("POST /ingest error: latency=%.2fms", elapsed * 1000)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    min_price: float = Query(None, ge=0, description="Minimum price filter"),
    max_price: float = Query(None, ge=0, description="Maximum price filter"),
    min_rating: float = Query(None, ge=0, le=5, description="Minimum rating filter (0-5)"),
    category: str = Query(None, min_length=1, max_length=200, description="Category filter (substring match)"),
):
    """
    Search products via contextual query expansion and FAISS vector search.
    - Expands query via HuggingFace LLM (with fallback to original)
    - Searches FAISS index for top-10 matches
    - Applies optional filters (price, rating, category)
    - Ranks by hybrid score: 70% semantic similarity + 30% engagement heuristic
    - Returns results with AI explanation and individual ranking scores
    """
    start_time = time.perf_counter()
    try:
        # Pre-checks for initialization
        if not os.path.exists(INDEX_PATH):
            raise HTTPException(status_code=400, detail="Vector index not initialized. Ingest data first.")

        def _db_has_products(db_path: str) -> bool:
            try:
                with sqlite3.connect(db_path) as conn:
                    # Check table exists
                    cur = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='products'")
                    if (cur.fetchone() or [0])[0] == 0:
                        return False
                    # Check rows exist
                    cur = conn.execute("SELECT COUNT(*) FROM products")
                    cnt = (cur.fetchone() or [0])[0]
                    return cnt > 0
            except Exception:
                return False

        if not os.path.exists(DB_PATH) or not _db_has_products(DB_PATH):
            raise HTTPException(status_code=400, detail="Database not initialized or empty. Ingest data first.")
        logger.info(
            "GET /search start: query='%s', filters=[price:%s-%s, rating:%s, category:%s]",
            query,
            min_price,
            max_price,
            min_rating,
            category,
        )
        result = search_service.search(
            query=query,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            category=category,
        )
        elapsed = time.perf_counter() - start_time
        result["latency_ms"] = round(elapsed * 1000, 2)
        logger.info(
            "GET /search success: results=%d, latency=%.2fms",
            result.get("total", 0),
            result["latency_ms"],
        )
        return JSONResponse(result)
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.exception("GET /search error: latency=%.2fms", elapsed * 1000)
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# ---------------------- Phase 3: Tracking & Learning ----------------------

class EventRequest(BaseModel):
    """Event tracking request payload."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    product_id: int = Field(..., gt=0, description="Product ID")
    type: str = Field(..., pattern="^(click|cart|purchase|bounce|dwell)$", description="Event type")
    value: float = Field(None, description="Optional value (e.g., dwell time in seconds)")


@app.post("/track")
async def track_event(event: EventRequest):
    """
    Track user engagement events (click, cart, purchase, bounce, dwell).
    - Asynchronously logs events to SQLite events table
    - Non-blocking insertion for low-latency response
    - Events feed into learning service for heuristic scoring
    
    Example:
    POST /track
    {
        "query": "wireless headphones",
        "product_id": 42,
        "type": "purchase",
        "value": null
    }
    """
    start_time = time.perf_counter()
    try:
        logger.info(
            "POST /track start: query='%s', product_id=%d, type=%s",
            event.query,
            event.product_id,
            event.type,
        )
        result = tracking_service.track_event(
            query=event.query,
            product_id=event.product_id,
            event_type=event.type,
            value=event.value,
            async_insert=True,  # Non-blocking
        )
        elapsed = time.perf_counter() - start_time
        result["latency_ms"] = round(elapsed * 1000, 2)
        logger.info(
            "POST /track success: status=%s, latency=%.2fms",
            result.get("status", "unknown"),
            result["latency_ms"],
        )
        return JSONResponse(result)
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.exception("POST /track error: latency=%.2fms", elapsed * 1000)
        raise HTTPException(status_code=500, detail=f"Tracking error: {str(e)}")


@app.post("/learn")
def learn_from_events(query: str = Query(None, description="Optional: recompute for specific query only")):
    """
    Trigger learning from accumulated events.
    - Aggregates clicks, purchases, bounces per (query, product_id)
    - Computes heuristic scores: score = 2*clicks + 5*purchases - bounces
    - Heuristic rationale:
      * Purchases (5x): Strongest signal of product-query fitness
      * Clicks (2x): Shows user interest and engagement
      * Bounces (1x penalty): Indicates poor match, discourages ranking
    - Stores scores in SQLite scores table for use in hybrid ranking
    - If query param is provided, recomputes only that query; else recomputes all
    
    Example:
    POST /learn?query=wireless%20headphones
    """
    start_time = time.perf_counter()
    try:
        logger.info("POST /learn start: query=%s", query)
        result = learning_service.learn_from_events(query=query)
        elapsed = time.perf_counter() - start_time
        result["latency_ms"] = round(elapsed * 1000, 2)
        logger.info(
            "POST /learn success: queries_processed=%d, scores_updated=%d, latency=%.2fms",
            result.get("queries_processed", 0),
            result.get("scores_updated", 0),
            result["latency_ms"],
        )
        return JSONResponse(result)
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.exception("POST /learn error: latency=%.2fms", elapsed * 1000)
        raise HTTPException(status_code=500, detail=f"Learning error: {str(e)}")
