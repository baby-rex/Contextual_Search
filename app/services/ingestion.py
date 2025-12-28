"""
Ingestion service for Phase 1.
- Reads uploaded CSV/JSON bytes via pandas
- Normalizes fields per spec
- Computes SentenceTransformers embeddings (all-MiniLM-L6-v2)
- Persists products in SQLite and vectors in FAISS IndexIDMap2(IndexFlatIP)
- Batch processing, logging, and robust error handling
"""

import io
import json
import logging
import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except ImportError as e:
    raise RuntimeError("faiss library is required. Please install faiss-cpu.") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise RuntimeError("sentence-transformers is required. Please install it.") from e

logger = logging.getLogger("ingestion")


class IngestionService:
    """Handles dataset ingestion, normalization, persistence, and vector indexing."""

    MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim
    EMBED_BATCH_SIZE = 128

    def __init__(self, db_path: str, index_path: str) -> None:
        self.db_path = db_path
        self.index_path = index_path
        self._model_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._db_lock = threading.Lock()

        # Prepare filesystem
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        # Ensure DB initialized
        self._init_db()

        # Lazy-load model and index
        self._model: SentenceTransformer | None = None
        self._index: Any | None = None

    # ---------------------- Public API ----------------------
    def ingest_bytes(self, content_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """
        Ingest dataset from raw bytes; supports CSV or JSON.
        Returns ingestion stats.
        """
        df = self._read_to_dataframe(content_bytes, filename, content_type)
        if df.empty:
            return {
                "inserted": 0,
                "skipped": 0,
                "index_size": self._get_index_size(),
                "message": "No rows to ingest",
            }

        # Normalize
        df_norm = self._normalize_dataframe(df)
        if df_norm.empty:
            return {
                "inserted": 0,
                "skipped": len(df),
                "index_size": self._get_index_size(),
                "message": "All rows invalid after normalization",
            }

        # Assign global incremental IDs starting from current max
        start_id = self._get_current_max_id() + 1
        df_norm["id"] = np.arange(start_id, start_id + len(df_norm))

        # Persist to DB
        inserted, skipped = self._persist_products(df_norm)

        # Embed and index
        ids = df_norm["id"].astype(np.int64).to_numpy()
        searchable_texts = df_norm["searchable_text"].astype(str).tolist()
        vectors = self._encode_texts(searchable_texts)
        self._add_to_index(ids, vectors)

        # Save index
        self._save_index()

        return {
            "inserted": inserted,
            "skipped": skipped,
            "index_size": self._get_index_size(),
        }

    # ---------------------- Data Reading ----------------------
    def _read_to_dataframe(self, content_bytes: bytes, filename: str, content_type: str) -> pd.DataFrame:
        """Read uploaded bytes into a DataFrame, supporting CSV and JSON."""
        name_lower = (filename or "").lower()
        ct_lower = (content_type or "").lower()
        buf = io.BytesIO(content_bytes)
        try:
            if name_lower.endswith(".csv") or "csv" in ct_lower:
                df = pd.read_csv(buf, encoding="utf-8", on_bad_lines="skip")
            elif name_lower.endswith(".json") or "json" in ct_lower:
                # Try standard JSON array; fallback to JSON Lines
                data = buf.getvalue().decode("utf-8")
                try:
                    parsed = json.loads(data)
                    df = pd.DataFrame(parsed)
                except json.JSONDecodeError:
                    df = pd.read_json(io.StringIO(data), lines=True)
            else:
                # Attempt CSV as general fallback
                logger.warning("Unknown content type; attempting CSV parse")
                buf.seek(0)
                df = pd.read_csv(buf, encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            logger.exception("Failed to parse uploaded file")
            raise ValueError(f"Failed to parse file: {e}")

        # Drop completely empty rows
        df = df.dropna(how="all")
        logger.info("Read dataframe with %d rows and %d columns", len(df), len(df.columns))
        return df

    # ---------------------- Normalization ----------------------
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize columns per spec:
        - title <- 'name'
        - category <- 'main_category' | 'sub_category'
        - price <- discount_price or actual_price (float, strip ₹/commas; defaults to 0.0)
        - rating <- 'ratings' float (defaults to 0.0)
        - rating count <- 'no_of_ratings' int (strip commas; defaults to 0)
        - description <- '' (optionally concat category)
        - attributes_json <- simple attributes parsed from title (+ rating_count if present)
        - searchable_text <- title + ' ' + category
        """
        cols = {c.lower(): c for c in df.columns}

        def get_col(*names: str) -> pd.Series:
            for n in names:
                if n in cols:
                    return df[cols[n]]
            return pd.Series([None] * len(df))

        name_series = get_col("name")
        main_cat_series = get_col("main_category")
        sub_cat_series = get_col("sub_category")
        discount_series = get_col("discount_price", "discountprice")
        actual_series = get_col("actual_price", "price")
        ratings_series = get_col("ratings", "rating")
        ratings_count_series = get_col("no_of_ratings", "num_ratings", "ratings_count", "number_of_ratings")

        def safe_float(val: Any, default: float = 0.0) -> float:
            """Convert value to float safely: strips currency, commas; returns default on failure."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            try:
                s = str(val).strip()
                if s == "" or s.lower() == "nan":
                    return default
                # Remove currency symbols and thousands separators
                s = s.replace("₹", "").replace(",", "")
                # Keep digits and dot only
                s = re.sub(r"[^0-9.]+", "", s)
                if s == "" or s == ".":
                    return default
                return float(s)
            except Exception:
                return default

        def safe_int(val: Any, default: int = 0) -> int:
            """Convert value to int safely: strips commas/currency; returns default on failure."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return default
            try:
                s = str(val).strip()
                if s == "" or s.lower() == "nan":
                    return default
                s = s.replace("₹", "").replace(",", "")
                # Remove non-digits
                s = re.sub(r"[^0-9]+", "", s)
                if s == "":
                    return default
                return int(s)
            except Exception:
                return default

        def make_title(x: Any) -> str:
            s = str(x).strip() if pd.notna(x) else ""
            return s

        def make_category(main: Any, sub: Any) -> str:
            m = str(main).strip() if pd.notna(main) else ""
            s = str(sub).strip() if pd.notna(sub) else ""
            if m and s:
                return f"{m} | {s}"
            return m or s

        def parse_price(discount: Any, actual: Any) -> float:
            # Prefer discount if present; else actual. Use safe defaults.
            val = discount if (pd.notna(discount) and str(discount).strip() != "") else actual
            return safe_float(val, default=0.0)

        def parse_rating(r: Any) -> float:
            return safe_float(r, default=0.0)

        def parse_attributes(title: str) -> Dict[str, Any]:
            """Simple heuristic attribute extraction from title (size, counts)."""
            attrs: Dict[str, Any] = {}
            t = title.lower()
            # pack of N
            m = re.search(r"pack\s*of\s*(\d+)", t)
            if m:
                attrs["pack_of"] = int(m.group(1))
            # size with units
            m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg|pcs|piece|pieces)", t)
            if m:
                val = float(m.group(1))
                unit = m.group(2)
                attrs["size"] = {"value": val, "unit": unit}
            # numbers
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", t)
            if nums:
                attrs.setdefault("numbers", [float(n) for n in nums])
            return attrs

        data = {
            "title": name_series.apply(make_title),
            "category": [
                make_category(m, s) for m, s in zip(main_cat_series.tolist(), sub_cat_series.tolist())
            ],
        }
        # Price and rating (defaults: price=0.0, rating=0.0)
        data["price"] = [
            parse_price(d, a) for d, a in zip(discount_series.tolist(), actual_series.tolist())
        ]
        data["rating"] = ratings_series.apply(parse_rating)

        # Ratings count (defaults: 0) – stored in attributes_json for safety without schema changes
        rating_counts = ratings_count_series.apply(lambda x: safe_int(x, default=0)).tolist()

        # Description (empty per spec; keep option to append category if needed)
        data["description"] = ["" for _ in range(len(df))]

        # Attributes JSON
        titles = data["title"].tolist()
        attrs_list = [parse_attributes(t) if t else {} for t in titles]
        # Attach rating_count if present
        for i, cnt in enumerate(rating_counts):
            if cnt and cnt > 0:
                try:
                    attrs_list[i]["rating_count"] = int(cnt)
                except Exception:
                    # Defensive: keep ingestion robust
                    pass
        data["attributes_json"] = [json.dumps(a, ensure_ascii=False) for a in attrs_list]

        # Searchable text
        cats = data["category"]
        data["searchable_text"] = [
            f"{t} {c}".strip() for t, c in zip(titles, cats)
        ]

        out = pd.DataFrame(data)
        # Remove rows without title
        out = out[out["title"].astype(str).str.strip() != ""]
        out.reset_index(drop=True, inplace=True)
        logger.info("Normalized dataframe to %d rows", len(out))
        return out

    # ---------------------- DB Persistence ----------------------
    def _init_db(self) -> None:
        """Create products table if not exists."""
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    price REAL,
                    rating REAL,
                    attributes_json TEXT
                )
                """
            )
            conn.commit()
        logger.info("SQLite initialized at %s", self.db_path)

    def _get_current_max_id(self) -> int:
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COALESCE(MAX(id), 0) FROM products")
            row = cur.fetchone()
            return int(row[0] or 0)

    def _persist_products(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Insert normalized products into SQLite.
        Returns (inserted_count, skipped_count).
        """
        rows = [
            (
                int(df.loc[i, "id"]),
                str(df.loc[i, "title"]),
                str(df.loc[i, "description"]),
                str(df.loc[i, "category"]),
                (float(df.loc[i, "price"]) if pd.notna(df.loc[i, "price"]) else None),
                (float(df.loc[i, "rating"]) if pd.notna(df.loc[i, "rating"]) else None),
                str(df.loc[i, "attributes_json"]),
            )
            for i in range(len(df))
        ]
        inserted = 0
        skipped = 0
        with self._db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            for r in rows:
                try:
                    conn.execute(
                        """
                        INSERT INTO products (id, title, description, category, price, rating, attributes_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        r,
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate id, skip
                    skipped += 1
                except Exception:
                    logger.exception("Failed to insert row with id=%s", r[0])
                    skipped += 1
            conn.commit()
        logger.info("Persisted %d rows (skipped %d)", inserted, skipped)
        return inserted, skipped

    # ---------------------- Embeddings and Index ----------------------
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    logger.info("Loading SentenceTransformer model: %s", self.MODEL_NAME)
                    self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _load_index(self, dim: int) -> Any:
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    if os.path.exists(self.index_path):
                        logger.info("Loading existing FAISS index: %s", self.index_path)
                        self._index = faiss.read_index(self.index_path)
                        # Ensure it's an IDMap2; if not, wrap
                        if not isinstance(self._index, faiss.IndexIDMap2):
                            self._index = faiss.IndexIDMap2(self._index)
                    else:
                        logger.info("Creating new FAISS IndexFlatIP(%d)", dim)
                        base = faiss.IndexFlatIP(dim)
                        self._index = faiss.IndexIDMap2(base)
        return self._index

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        # encode returns numpy array; use batch for speed
        vectors = model.encode(
            texts,
            batch_size=self.EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        )
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors)
        return vectors.astype("float32")

    def _add_to_index(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("IDs and vectors length mismatch")
        dim = vectors.shape[1]
        index = self._load_index(dim)
        with self._index_lock:
            # Add with IDs
            index.add_with_ids(vectors, ids)
        logger.info("Added %d vectors to FAISS", len(ids))

    def _save_index(self) -> None:
        with self._index_lock:
            if self._index is None:
                return
            faiss.write_index(self._index, self.index_path)
        logger.info("FAISS index saved to %s", self.index_path)

    def _get_index_size(self) -> int:
        if self._index is None and os.path.exists(self.index_path):
            try:
                self._index = faiss.read_index(self.index_path)
            except Exception:
                return 0
        if self._index is None:
            return 0
        try:
            return int(self._index.ntotal)
        except Exception:
            return 0
