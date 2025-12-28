"""
Learning service for Phase 3.
- Aggregate events from tracking.events table
- Compute heuristic engagement scores via SQL
- Persist scores in SQLite scores table
- Heuristic formula: score = 2*clicks + 5*purchases - bounces
  Rationale:
  - Purchases indicate strong intent/satisfaction (highest weight: 5)
  - Clicks show user interest (medium weight: 2)
  - Bounces indicate poor match (penalty: -1)
  - This encourages the ranking to promote products with proven conversions
  - Scores complement vector similarity for hybrid ranking
"""

import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("learning")


class LearningService:
    """Computes and manages heuristic engagement scores from event data."""

    def __init__(self, db_path: str) -> None:
        """
        Initialize LearningService.
        
        Args:
            db_path: Path to SQLite products.db (shared with Phase 1 and 3)
        """
        self.db_path = db_path
        self._db_lock = threading.Lock()
        self._init_db()

    # ---------------------- Database Setup ----------------------
    def _init_db(self) -> None:
        """Create scores table if not exists."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    query TEXT NOT NULL,
                    product_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    clicks INTEGER DEFAULT 0,
                    purchases INTEGER DEFAULT 0,
                    bounces INTEGER DEFAULT 0,
                    PRIMARY KEY (query, product_id)
                )
                """
            )
            conn.commit()
        logger.info("Scores table initialized")

    # ---------------------- Heuristic Scoring ----------------------
    def learn_from_events(self, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Aggregate events and compute heuristic scores.
        
        Heuristic formula:
            score = 2 * clicks + 5 * purchases - bounces
        
        This weighting:
        - Prioritizes purchases (5x multiplier) as strongest signal of product fitness
        - Counts clicks (2x multiplier) as secondary signal of interest
        - Penalizes bounces (1x penalty) as sign of poor match
        - Encourages ranking to surface products with proven engagement
        
        Args:
            query: If specified, recompute scores only for this query;
                   if None, recompute all queries
        
        Returns: {
            'queries_processed': int,
            'scores_updated': int,
            'scores_deleted': int,
            'sample_scores': [{query, product_id, score, clicks, purchases, bounces}, ...]
        }
        """
        try:
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                # Step 1: Get unique (query, product_id) pairs from events
                if query:
                    where_clause = "WHERE events.query = ?"
                    params = (query,)
                else:
                    where_clause = ""
                    params = ()

                # Step 2: Aggregate event counts
                aggregation_query = f"""
                    SELECT
                        events.query,
                        events.product_id,
                        SUM(CASE WHEN events.type = 'click' THEN 1 ELSE 0 END) as clicks,
                        SUM(CASE WHEN events.type = 'cart' THEN 1 ELSE 0 END) as carts,
                        SUM(CASE WHEN events.type = 'purchase' THEN 1 ELSE 0 END) as purchases,
                        SUM(CASE WHEN events.type = 'bounce' THEN 1 ELSE 0 END) as bounces
                    FROM events
                    {where_clause}
                    GROUP BY events.query, events.product_id
                """
                cur = conn.execute(aggregation_query, params)
                rows = cur.fetchall()

                # Step 3: Compute heuristic scores and insert/update in scores table
                scores_updated = 0
                for row in rows:
                    q, pid, clicks, carts, purchases, bounces = row
                    clicks = clicks or 0
                    purchases = purchases or 0
                    bounces = bounces or 0

                    # Heuristic: 2*clicks + 5*purchases - bounces
                    heuristic_score = 2 * clicks + 5 * purchases - bounces

                    conn.execute(
                        """
                        INSERT INTO scores (query, product_id, score, clicks, purchases, bounces)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(query, product_id)
                        DO UPDATE SET
                            score = excluded.score,
                            clicks = excluded.clicks,
                            purchases = excluded.purchases,
                            bounces = excluded.bounces
                        """,
                        (q, pid, heuristic_score, clicks, purchases, bounces),
                    )
                    scores_updated += 1

                conn.commit()

                # Step 4: Get sample scores for response
                sample_cur = conn.execute(
                    """
                    SELECT query, product_id, score, clicks, purchases, bounces
                    FROM scores
                    ORDER BY score DESC
                    LIMIT 10
                    """
                )
                samples = [
                    {
                        "query": row[0],
                        "product_id": row[1],
                        "score": row[2],
                        "clicks": row[3],
                        "purchases": row[4],
                        "bounces": row[5],
                    }
                    for row in sample_cur.fetchall()
                ]

            logger.info("Learning complete: processed %d scores", scores_updated)
            return {
                "queries_processed": len(rows),
                "scores_updated": scores_updated,
                "sample_scores": samples,
            }

        except Exception as e:
            logger.exception("Learning from events failed")
            return {"error": str(e)}

    # ---------------------- Score Retrieval ----------------------
    def get_score(self, query: str, product_id: int) -> float:
        """
        Get heuristic score for a query-product pair.
        Returns 0.0 if not found.
        """
        try:
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT score FROM scores WHERE query = ? AND product_id = ?",
                    (query, product_id),
                )
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
        except Exception as e:
            logger.warning("Failed to get score for query=%s, product_id=%d: %s", query, product_id, e)
            return 0.0

    def get_top_scores_for_query(self, query: str, limit: int = 100) -> Dict[int, float]:
        """
        Get heuristic scores for all products in a query, ranked by score.
        Returns {product_id: score, ...}.
        """
        try:
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """
                    SELECT product_id, score FROM scores
                    WHERE query = ?
                    ORDER BY score DESC
                    LIMIT ?
                    """,
                    (query, limit),
                )
                return {row[0]: float(row[1]) for row in cur.fetchall()}
        except Exception as e:
            logger.warning("Failed to get top scores for query=%s: %s", query, e)
            return {}

    # ---------------------- Normalization Helper ----------------------
    @staticmethod
    def normalize_heuristic_score(raw_score: float, max_reference: float = 100.0) -> float:
        """
        Normalize heuristic score to [0, 1] for blending with similarity scores.
        
        Uses sigmoid-like scaling: normalized = raw_score / (max_reference + raw_score)
        This ensures:
        - Score of 0 -> 0.0
        - Score of max_reference -> 0.5
        - Score of 2*max_reference -> 0.67
        - Unbounded scores naturally cap toward 1.0
        """
        if raw_score < 0:
            # Negative scores (heavy bounces) still contribute negatively but capped
            return max(raw_score / (max_reference - raw_score), -1.0) if raw_score > -max_reference else -1.0
        return raw_score / (max_reference + raw_score)
