"""
Search service for Phase 2.
- Query expansion via HuggingFace LLM (gpt2 or better) with fallback to original query
- FAISS vector search (top-k) with cosine similarity scoring
- SQLite metadata fetch and filtering (price, rating, category)
- Result ranking and AI explanation
- Modular, robust error handling, logging
"""

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import numpy as np

# Lazy import to avoid circular dependency
_learning_service = None

try:
    import faiss  # type: ignore
except ImportError as e:
    raise RuntimeError("faiss library is required.") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise RuntimeError("sentence-transformers is required.") from e

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    raise RuntimeError("transformers library is required.") from e

logger = logging.getLogger("search")

# Feature flag: Set to False to disable LLM query expansion (saves memory on local machines)
USE_LLM = False


class SearchService:
    """
    Handles query expansion, embedding, FAISS search, and result ranking.
    Integrates with Phase 1 DB and index.
    """

    EMBED_MODEL = "all-MiniLM-L6-v2"
    LLM_TIMEOUT_SECONDS = 5
    FAISS_TOP_K = 10
    LLM_MAX_NEW_TOKENS = 20

    def __init__(self, db_path: str, index_path: str, hf_token: Optional[str] = None) -> None:
        """
        Initialize SearchService.
        
        Args:
            db_path: Path to SQLite products.db
            index_path: Path to FAISS index
            hf_token: Optional HuggingFace API token for LLM expansion
        """
        self.db_path = db_path
        self.index_path = index_path
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        # Thread-safe caches
        self._model_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._llm_lock = threading.Lock()

        self._embed_model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[Any] = None
        self._llm_pipeline: Optional[Any] = None
        self._llm_name: Optional[str] = None
        self._learning_service: Optional[Any] = None

    # ---------------------- Query Expansion (LLM) ----------------------
    def _load_llm_pipeline(self) -> Optional[Any]:
        """
        Lazy-load LLM pipeline for query expansion.
        Tries gpt2 or transformers-based model.
        Returns None if loading fails (will fall back to original query).
        """
        if self._llm_pipeline is not None:
            return self._llm_pipeline

        with self._llm_lock:
            if self._llm_pipeline is not None:
                return self._llm_pipeline

            try:
                # Attempt to load gpt2-based text generation
                logger.info("Loading LLM pipeline for query expansion")
                self._llm_pipeline = pipeline(
                    "text-generation",
                    model="gpt2",  # fallback to gpt2; can be upgraded
                    device=-1,  # CPU; set to 0 for GPU
                    max_new_tokens=self.LLM_MAX_NEW_TOKENS,
                )
                self._llm_name = "gpt2"
                logger.info("LLM pipeline loaded: %s", self._llm_name)
                return self._llm_pipeline
            except Exception as e:
                logger.warning("Failed to load LLM pipeline: %s. Query expansion disabled.", e)
                self._llm_pipeline = False  # Mark as failed to avoid retries
                return None

    def _expand_query(self, query: str) -> tuple[str, str]:
        """
        Expand query using LLM with timeout and fallback.
        Returns (expanded_query, explanation_text).
        
        If LLM fails or times out, returns (original_query, "Using original query due to expansion unavailable").
        """
        original = query.strip()
        llm = self._load_llm_pipeline()

        if llm is None or llm is False:
            return original, "Using original query (LLM unavailable)"

        try:
            # Prompt for synonyms/expansion
            prompt = f"Expand search query with synonyms: {original}"

            # Call with timeout via threading
            import threading
            result_holder = {"expanded": None, "error": None}

            def run_expansion():
                try:
                    response = llm(prompt, max_length=100, do_sample=True)
                    if response and len(response) > 0:
                        expanded_text = response[0].get("generated_text", "").replace(prompt, "").strip()
                        result_holder["expanded"] = expanded_text if expanded_text else original
                    else:
                        result_holder["expanded"] = original
                except Exception as e:
                    result_holder["error"] = str(e)
                    result_holder["expanded"] = original

            thread = threading.Thread(target=run_expansion, daemon=False)
            thread.start()
            thread.join(timeout=self.LLM_TIMEOUT_SECONDS)

            if thread.is_alive():
                logger.warning("LLM expansion timed out after %ds", self.LLM_TIMEOUT_SECONDS)
                return original, "Query expansion timed out; using original query"

            if result_holder["error"]:
                logger.warning("LLM expansion error: %s", result_holder["error"])
                return original, f"Query expansion failed: {result_holder['error']}"

            expanded = result_holder["expanded"]
            # Combine original + expanded for robust search
            combined = f"{original} {expanded}".strip()
            explanation = f"Expanded query to: {expanded}"
            logger.info("Query expanded from '%s' to '%s'", original, combined)
            return combined, explanation

        except Exception as e:
            logger.exception("Unexpected error during query expansion")
            return original, f"Query expansion error: {str(e)}"

    # ---------------------- Embedding ----------------------
    def _load_embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            with self._model_lock:
                if self._embed_model is None:
                    logger.info("Loading embedding model: %s", self.EMBED_MODEL)
                    self._embed_model = SentenceTransformer(self.EMBED_MODEL)
        return self._embed_model

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query text. Returns float32 vector."""
        model = self._load_embed_model()
        vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        return np.array(vec, dtype="float32").reshape(1, -1)

    # ---------------------- FAISS Search ----------------------
    def _load_faiss_index(self) -> Optional[Any]:
        """Lazy-load FAISS index. Returns None if not found or error."""
        if self._faiss_index is not None:
            return self._faiss_index

        with self._index_lock:
            if self._faiss_index is not None:
                return self._faiss_index

            if not os.path.exists(self.index_path):
                logger.warning("FAISS index not found at %s", self.index_path)
                self._faiss_index = False
                return None

            try:
                logger.info("Loading FAISS index from %s", self.index_path)
                index = faiss.read_index(self.index_path)
                self._faiss_index = index
                return index
            except Exception as e:
                logger.exception("Failed to load FAISS index")
                self._faiss_index = False
                return None

    def _search_faiss(self, query_vector: np.ndarray, k: int = FAISS_TOP_K) -> Dict[str, Any]:
        """
        Search FAISS index.
        Returns {'ids': [...], 'distances': [...], 'count': int}.
        Distances are inner product scores; normalize to [0, 1] cosine similarity.
        """
        index = self._load_faiss_index()
        if index is None or index is False:
            logger.error("FAISS index not available")
            return {"ids": [], "distances": [], "count": 0}

        try:
            with self._index_lock:
                distances, ids = index.search(query_vector, k)
            # distances shape: (1, k) for single query
            distances = distances[0]  # Extract 1D array
            ids = ids[0].astype(int)

            # Convert inner product scores to [0, 1] via (score + 1) / 2 normalization
            # since vectors are L2-normalized, IP is in [-1, 1]
            similarities = (distances + 1.0) / 2.0
            similarities = np.clip(similarities, 0.0, 1.0)

            logger.info("FAISS search returned %d results", len(ids))
            return {"ids": ids.tolist(), "distances": similarities.tolist(), "count": len(ids)}
        except Exception as e:
            logger.exception("FAISS search failed")
            return {"ids": [], "distances": [], "count": 0}

    # ---------------------- SQLite Filtering ----------------------
    def _fetch_products(self, product_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch product metadata from SQLite by IDs. Returns {id: {...}}."""
        if not product_ids:
            return {}

        placeholders = ",".join("?" * len(product_ids))
        query = f"""
            SELECT id, title, description, category, price, rating, attributes_json
            FROM products
            WHERE id IN ({placeholders})
        """
        products = {}
        try:
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(query, product_ids)
                for row in cur:
                    pid = row["id"]
                    attrs = {}
                    try:
                        attrs = json.loads(row["attributes_json"] or "{}")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    products[pid] = {
                        "id": pid,
                        "title": row["title"],
                        "description": row["description"],
                        "category": row["category"],
                        "price": row["price"],
                        "rating": row["rating"],
                        "attributes": attrs,
                    }
            logger.info("Fetched %d products from SQLite", len(products))
        except Exception as e:
            logger.exception("Failed to fetch products from SQLite")
        return products

    def _apply_filters(
        self,
        products: Dict[int, Dict[str, Any]],
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        category: Optional[str] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Filter products by criteria.
        Returns filtered {id: {...}}.
        """
        filtered = {}
        for pid, prod in products.items():
            # Price filters
            if min_price is not None and (prod["price"] is None or prod["price"] < min_price):
                continue
            if max_price is not None and (prod["price"] is None or prod["price"] > max_price):
                continue
            # Rating filter
            if min_rating is not None and (prod["rating"] is None or prod["rating"] < min_rating):
                continue
            # Category filter (case-insensitive substring match)
            if category is not None:
                cat = prod["category"] or ""
                if category.lower() not in cat.lower():
                    continue
            filtered[pid] = prod
        logger.info("Applied filters: %d -> %d products", len(products), len(filtered))
        return filtered

    # ---------------------- Public Search API ----------------------
    def search(
        self,
        query: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute search:
        1. Expand query with LLM (with fallback)
        2. Embed expanded query
        3. FAISS top-k search
        4. Fetch products from SQLite
        5. Apply filters
        6. Rank by hybrid score: 0.7 * similarity + 0.3 * normalized_heuristic
        7. Return results with AI explanation
        
        Hybrid ranking formula:
        final_score = 0.7 * cosine_similarity + 0.3 * normalized_heuristic_score
        
        Rationale:
        - Cosine similarity (0.7 weight): Captures semantic relevance; stable and proven
        - Normalized heuristic (0.3 weight): Incorporates user engagement signals
          (clicks, purchases, bounces) to promote products users actually want
        - The 0.7/0.3 split ensures semantic relevance remains dominant while allowing
          engagement patterns to influence ranking, avoiding cold-start bias
        
        Returns {
            'query': original_query,
            'expanded_query': combined_expanded,
            'explanation': AI explanation,
            'results': [{'id', 'title', 'category', 'price', 'rating', 'similarity_score', 'heuristic_score', 'final_score', ...}, ...],
            'total': count,
            'filters_applied': {...}
        }
        """
        logger.info("Search initiated for query: '%s'", query)

        # Step 1: Expand query (guarded by USE_LLM flag)
        if USE_LLM:
            expanded_query, expansion_explanation = self._expand_query(query)
        else:
            expanded_query = query.strip()
            expansion_explanation = "LLM query expansion disabled; using original query"

        # Step 2: Embed
        query_vector = self._embed_query(expanded_query)

        # Step 3: FAISS search
        faiss_result = self._search_faiss(query_vector, k=self.FAISS_TOP_K)
        if faiss_result["count"] == 0:
            logger.info("No FAISS results found")
            return {
                "query": query,
                "expanded_query": expanded_query,
                "explanation": expansion_explanation + "; No matches found in index.",
                "results": [],
                "total": 0,
                "filters_applied": {
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_rating": min_rating,
                    "category": category,
                },
            }

        # Step 4: Fetch products
        products = self._fetch_products(faiss_result["ids"])
        if not products:
            logger.info("No products found in database")
            return {
                "query": query,
                "expanded_query": expanded_query,
                "explanation": expansion_explanation + "; Products not found in database.",
                "results": [],
                "total": 0,
                "filters_applied": {
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_rating": min_rating,
                    "category": category,
                },
            }

        # Step 5: Apply filters
        filtered = self._apply_filters(products, min_price, max_price, min_rating, category)

        # Step 6: Rank by hybrid score
        results = []
        id_to_similarity = {pid: score for pid, score in zip(faiss_result["ids"], faiss_result["distances"])}
        
        # Load learning service (lazy)
        learning_svc = self._get_learning_service()
        
        for pid, prod in filtered.items():
            sim_score = id_to_similarity.get(pid, 0.0)
            
            # Get heuristic score (engagement-based)
            raw_heuristic = 0.0
            if learning_svc:
                raw_heuristic = learning_svc.get_score(query, pid)
            
            # Normalize heuristic to [0, 1]
            if learning_svc:
                norm_heuristic = learning_svc.normalize_heuristic_score(raw_heuristic)
            else:
                norm_heuristic = 0.0
            
            # Hybrid ranking: 70% similarity + 30% engagement
            final_score = 0.7 * sim_score + 0.3 * max(0.0, norm_heuristic)
            
            results.append(
                {
                    "id": prod["id"],
                    "title": prod["title"],
                    "category": prod["category"],
                    "description": prod["description"],
                    "price": prod["price"],
                    "rating": prod["rating"],
                    "attributes": prod["attributes"],
                    "similarity_score": round(float(sim_score), 4),
                    "heuristic_score": round(float(raw_heuristic), 4),
                    "final_score": round(float(final_score), 4),
                }
            )
        # Sort by final score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info("Search completed: %d results after filtering and ranking", len(results))

        # Build final explanation
        final_explanation = expansion_explanation
        if len(filtered) < len(products):
            final_explanation += f"; Filtered {len(products) - len(filtered)} results by price/rating/category."
        if len(results) == 0 and len(filtered) > 0:
            final_explanation += " No results after filtering."
        if learning_svc:
            final_explanation += " (Ranked by 70% semantic + 30% engagement signals)"

        return {
            "query": query,
            "expanded_query": expanded_query,
            "explanation": final_explanation,
            "results": results,
            "total": len(results),
            "filters_applied": {
                "min_price": min_price,
                "max_price": max_price,
                "min_rating": min_rating,
                "category": category,
            },
        }

    def _get_learning_service(self) -> Optional[Any]:
        """Lazy-load LearningService to avoid circular imports."""
        if self._learning_service is None:
            try:
                # Import here to avoid circular dependency
                from app.services.learning import LearningService
                self._learning_service = LearningService(db_path=self.db_path)
            except Exception as e:
                logger.warning("Failed to load LearningService: %s", e)
                self._learning_service = False
        return self._learning_service if self._learning_service is not False else None
