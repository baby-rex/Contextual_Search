"""
Tracking service for Phase 3.
- Async event logging via threading to SQLite events table
- Supports event types: click, cart, purchase, bounce, dwell
- Non-blocking insertion for low-latency response
- Thread-safe database access
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("tracking")


class TrackingService:
    """Handles event tracking with async threaded insertion."""

    def __init__(self, db_path: str) -> None:
        """
        Initialize TrackingService.
        
        Args:
            db_path: Path to SQLite products.db (shared with Phase 1)
        """
        self.db_path = db_path
        self._db_lock = threading.Lock()
        self._init_db()

    # ---------------------- Database Setup ----------------------
    def _init_db(self) -> None:
        """Create events table if not exists."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    product_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    value REAL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            # Create index for fast event aggregation
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_query_product ON events(query, product_id)"
            )
            conn.commit()
        logger.info("Events table initialized")

    # ---------------------- Event Tracking ----------------------
    def track_event(
        self,
        query: str,
        product_id: int,
        event_type: str,
        value: Optional[float] = None,
        async_insert: bool = True,
    ) -> Dict[str, Any]:
        """
        Track an event (click, cart, purchase, bounce, dwell).
        
        Args:
            query: Original search query
            product_id: Product ID from result
            event_type: One of ['click', 'cart', 'purchase', 'bounce', 'dwell']
            value: Optional numeric value (e.g., dwell time in seconds)
            async_insert: If True, insert via background thread (non-blocking)
        
        Returns: {'event_id': int, 'status': 'logged'|'queued'} or error dict
        """
        if event_type not in ["click", "cart", "purchase", "bounce", "dwell"]:
            logger.warning("Invalid event type: %s", event_type)
            return {"error": f"Invalid event type: {event_type}"}

        timestamp = datetime.utcnow().isoformat()

        if async_insert:
            # Non-blocking: queue insertion in background thread
            thread = threading.Thread(
                target=self._insert_event_threaded,
                args=(query, product_id, event_type, value, timestamp),
                daemon=True,
            )
            thread.start()
            return {"status": "queued", "message": "Event logged asynchronously"}
        else:
            # Synchronous insert
            event_id = self._insert_event_sync(query, product_id, event_type, value, timestamp)
            if event_id:
                return {"event_id": event_id, "status": "logged"}
            else:
                return {"error": "Failed to insert event"}

    def _insert_event_sync(
        self,
        query: str,
        product_id: int,
        event_type: str,
        value: Optional[float],
        timestamp: str,
    ) -> Optional[int]:
        """Synchronous event insertion. Returns event_id or None."""
        try:
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO events (query, product_id, type, value, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (query, product_id, event_type, value, timestamp),
                )
                event_id = cur.lastrowid
                conn.commit()
            logger.info(
                "Event logged: query='%s', product_id=%d, type=%s, event_id=%d",
                query,
                product_id,
                event_type,
                event_id,
            )
            return event_id
        except Exception as e:
            logger.exception("Failed to insert event")
            return None

    def _insert_event_threaded(
        self,
        query: str,
        product_id: int,
        event_type: str,
        value: Optional[float],
        timestamp: str,
    ) -> None:
        """Background thread target for event insertion."""
        self._insert_event_sync(query, product_id, event_type, value, timestamp)

    # ---------------------- Event Querying ----------------------
    def get_event_stats(self, query: str, product_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get event statistics for a query or query+product pair.
        
        Returns:
        {
            'query': str,
            'product_id': int | None,
            'events': {
                'click': int,
                'cart': int,
                'purchase': int,
                'bounce': int,
                'dwell': {'count': int, 'total_time': float, 'avg_time': float}
            },
            'total_events': int
        }
        """
        try:
            with self._db_lock, sqlite3.connect(self.db_path) as conn:
                if product_id is not None:
                    # Query + product-specific
                    cur = conn.execute(
                        "SELECT type, COUNT(*) as cnt, SUM(value) as total_val FROM events "
                        "WHERE query = ? AND product_id = ? GROUP BY type",
                        (query, product_id),
                    )
                else:
                    # Query-wide stats
                    cur = conn.execute(
                        "SELECT type, COUNT(*) as cnt, SUM(value) as total_val FROM events "
                        "WHERE query = ? GROUP BY type",
                        (query,),
                    )
                rows = cur.fetchall()

            stats: Dict[str, Any] = {
                "click": 0,
                "cart": 0,
                "purchase": 0,
                "bounce": 0,
                "dwell": {"count": 0, "total_time": 0.0, "avg_time": 0.0},
            }
            total = 0
            for row in rows:
                evt_type, cnt, total_val = row
                if evt_type == "dwell":
                    stats["dwell"]["count"] = cnt
                    stats["dwell"]["total_time"] = float(total_val or 0)
                    stats["dwell"]["avg_time"] = (
                        stats["dwell"]["total_time"] / cnt if cnt > 0 else 0
                    )
                else:
                    stats[evt_type] = cnt
                total += cnt

            result = {
                "query": query,
                "product_id": product_id,
                "events": stats,
                "total_events": total,
            }
            logger.info("Event stats retrieved: %s", result)
            return result
        except Exception as e:
            logger.exception("Failed to get event stats")
            return {
                "query": query,
                "product_id": product_id,
                "events": {},
                "error": str(e),
            }
