# AI-Powered Contextual Search Platform

## Project Overview

This system solves the problem of delivering semantically relevant product search results while adapting to user behavior. It combines vector-based semantic search with engagement-based learning to rank products by both relevance and proven conversion signals. The platform ingests CSV product catalogs, generates embeddings, performs fast similarity search, tracks user interactions, and continuously improves ranking through behavioral feedback.

---

## System Architecture

The platform follows a modular layered design:

- **FastAPI Application** (`app/main.py`): Exposes 5 REST endpoints (`/health`, `/ingest`, `/search`, `/track`, `/learn`)
- **Service Layer**:
  - **IngestionService**: Parses CSV/JSON, normalizes data, generates embeddings, persists to SQLite and FAISS
  - **SearchService**: Handles query expansion, embedding, FAISS vector search, SQL filtering, and hybrid ranking
  - **TrackingService**: Asynchronously logs user interaction events
  - **LearningService**: Aggregates events and computes engagement-based scores
- **Persistence Layer**:
  - SQLite database (`db/products.db`) with tables: `products`, `events`, `scores`
  - FAISS vector index (`indexes/faiss.index`)

For architecture visualization, refer to the external architecture diagram.

---

## Data Flow

**Ingestion Phase:**
1. User uploads CSV/JSON via `/ingest`
2. IngestionService normalizes fields (title, category, price, rating, searchable_text)
3. SentenceTransformers generates 384-dim embeddings
4. Data stored in SQLite `products` table
5. Embeddings added to FAISS index

**Search Phase:**
1. User submits query to `/search` with optional filters
2. SearchService optionally expands query via LLM (feature-flagged; disabled locally)
3. Query embedded to 384-dim vector
4. FAISS performs top-k similarity search
5. Products fetched from SQLite by IDs
6. SQL filters applied (price, rating, category)
7. Results ranked using hybrid score: `0.7 * semantic_similarity + 0.3 * engagement_score`
8. AI-generated explanation returned with results

**Tracking Phase:**
1. User interactions (click, purchase, bounce, dwell) sent to `/track`
2. TrackingService asynchronously inserts events into SQLite `events` table

**Learning Phase:**
1. Admin/scheduler triggers `/learn`
2. LearningService aggregates events per (query, product_id)
3. Heuristic scores computed: `score = 2*clicks + 5*purchases - bounces`
4. Scores normalized and stored in SQLite `scores` table
5. Future searches use updated scores in hybrid ranking

---

## AI Usage

**1. Embeddings:**
- Uses SentenceTransformers model `all-MiniLM-L6-v2` to convert product text and queries into 384-dimensional vectors

**2. FAISS Semantic Search:**
- IndexFlatIP with L2-normalized vectors for cosine similarity
- Returns top-k most semantically similar products in real-time

**3. Automatic Attribute Extraction:**
- Lightweight heuristics parse product titles to extract structured attributes (e.g., pack size, units)
- Stored in `attributes_json` for filtering and enrichment

**4. AI-Based Re-Ranking:**
- Hybrid scoring combines:
  - **Semantic similarity (70%)**: Measures text relevance
  - **Engagement heuristic (30%)**: Incorporates user behavior signals
- Products with proven conversion signals rank higher

**5. AI-Generated Explanations:**
- Search results include explanations describing:
  - Query expansion details
  - Applied filters
  - Ranking methodology
- Provides transparency into why specific results were returned

**6. LLM Query Expansion:**
- Optional feature using HuggingFace `gpt2` model
- Expands queries with synonyms and related terms
- Currently disabled locally (feature flag: `USE_LLM = False`) to avoid memory issues
- Can be enabled in production environments

---

## Learning Logic

**Event Types Tracked:**
- `click`: User clicked on a product
- `purchase`: User purchased a product
- `bounce`: User quickly left after viewing
- `dwell`: User spent time on product page (value = seconds)

**Scoring Formula:**
```
heuristic_score = 2 × clicks + 5 × purchases - 1 × bounces
```

**Rationale:**
- **Purchases (5x)**: Strongest signal of product-query fitness
- **Clicks (2x)**: Indicates user interest
- **Bounces (−1x)**: Penalizes poor matches

**Ranking Update Mechanism:**
1. `/learn` aggregates events from `events` table
2. Computes raw heuristic scores
3. Normalizes scores to [0, 1] range
4. Stores in `scores` table keyed by (query, product_id)
5. Next search retrieves scores and blends: `final_score = 0.7*similarity + 0.3*normalized_heuristic`
6. Products with higher engagement scores rank better over time

---

## API Overview

- **`GET /health`**: Returns service status
- **`POST /ingest`**: Accepts CSV/JSON file upload; normalizes, embeds, and stores products
- **`GET /search`**: Performs semantic search with optional filters; returns ranked results with explanations
- **`POST /track`**: Logs user interaction events asynchronously
- **`POST /learn`**: Aggregates events and updates engagement-based scores

---

## How to Run

**Local Execution (Python):**
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --port 8000

# Access API
# Swagger UI: http://localhost:8000/docs
```

**Docker Execution:**
```bash
# Build image
docker build -t contextual-search .

# Run container
docker run --rm -p 8000:8000 contextual-search

# Access API at http://localhost:8000/docs
```

---

## Sample Dataset

- Includes 1-2 representative CSV files (`sample_products.csv`) for testing ingestion
- Format: Amazon product catalog schema (name, main_category, sub_category, discount_price, actual_price, ratings, no_of_ratings)
- Larger datasets (1K+ products) were used during development but excluded from repository due to size constraints
- System supports any CSV with similar schema; robust sanitization handles missing values and currency symbols

---

## Summary

This platform delivers production-ready contextual search with:
- **Semantic understanding** via SentenceTransformers and FAISS vector search
- **Behavioral learning** from user interactions to improve ranking
- **Hybrid ranking** balancing relevance (70%) and engagement (30%)
- **Modular architecture** with clear separation of concerns
- **Robust data handling** for noisy real-world CSV files
- **Optional LLM expansion** available for production environments
