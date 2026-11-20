"""
================================================================================
DreamTrip AI — Redis Cache & Semantic Token Conservation Subsystem
================================================================================
This module provides an enterprise-grade semantic caching layer:
1. Exact & Syntactic Caching: SHA-256 hash matching on normalized prompts (< 1ms).
2. Canonical Intent Caching: Matches on extracted travel parameters (destination,
   duration, budget) so rephrased queries share the exact same cached plan.
3. Semantic Vector Similarity: Dense vector embeddings with Cosine Similarity
   thresholding (>= 0.88) via Gemini 'text-embedding-004' (with deterministic
   hashing vectorizer fallback).
4. Automated Failover: High-speed in-memory fallback if Redis is offline.
5. Telemetry Tracking: Hits, misses, semantic vs exact hits, and cumulative
   conserved tokens.
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. Standard Library & System Utilities
# ------------------------------------------------------------------------------
import os
import json
import hashlib
import time
import re
import math
from typing import Tuple, List, Dict, Any, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional HTTP requests for Gemini embedding API
try:
    import requests
except ImportError:
    requests = None

# ------------------------------------------------------------------------------
# 2. Redis Client Import with Safe Fallback
# ------------------------------------------------------------------------------
try:
    import redis
except ImportError:
    redis = None

# Default Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# In-memory dictionary fallback structures
_MEMORY_CACHE: Dict[str, Any] = {}       # { key: payload }
_MEMORY_EXPIRY: Dict[str, float] = {}     # { key: unix_timestamp }
_SEMANTIC_INDEX: List[Dict[str, Any]] = [] # [ { "text": str, "vector": list[float], "key": str, "expiry": float } ]


# ==============================================================================
# Vector & Cosine Similarity Utilities
# ==============================================================================
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two numeric vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_embedding(text: str) -> List[float]:
    """
    Fetches dense vector embeddings for text:
    1. Primary: Google Gemini 'text-embedding-004' via REST API.
    2. Fallback: Deterministic 128-dimensional character n-gram hashing vectorizer.
    """
    clean_text = text.strip()
    if not clean_text:
        return [0.0] * 128

    # 1. Try Google Gemini text-embedding-004
    if requests and GEMINI_API_KEY:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
            resp = requests.post(
                url,
                json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": clean_text[:1000]}]}
                },
                timeout=3
            )
            if resp.status_code == 200:
                values = resp.json().get("embedding", {}).get("values")
                if values and isinstance(values, list):
                    return values
        except Exception:
            pass  # Fall back to hashing vectorizer

    # 2. Deterministic Subword & N-Gram Hashing Vectorizer (Zero-dependency fallback)
    dim = 128
    vector = [0.0] * dim
    words = re.findall(r"\w+", clean_text.lower())

    for word in words:
        # Word-level hash
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vector[idx] += 1.0
        # Character 3-grams
        if len(word) >= 3:
            for i in range(len(word) - 2):
                ngram = word[i:i+3]
                n_idx = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16) % dim
                vector[n_idx] += 0.5

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


# ==============================================================================
# Redis Cache Manager with Semantic Caching Engine
# ==============================================================================
class RedisCacheManager:
    """
    Unified Caching Engine supporting:
    - Syntactic SHA-256 caching
    - Canonical Intent-based caching
    - Semantic Vector Similarity caching (threshold >= 0.88)
    - Redis + In-Memory Fallback
    """
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.client = None
        self.is_connected = False
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self._init_connection()

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    def _init_connection(self):
        """Initializes Redis connection with automatic docker host fallback."""
        if not redis:
            print("[RedisCache] redis package not available. Using in-memory fallback.")
            return

        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3
            )
            self.client.ping()
            self.is_connected = True
            print(f"[RedisCache] Successfully connected to Redis at {self.redis_url}")
            self._load_semantic_index_from_redis()
        except Exception as e:
            if "localhost" in self.redis_url or "127.0.0.1" in self.redis_url:
                docker_host_url = self.redis_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
                try:
                    docker_client = redis.from_url(
                        docker_host_url,
                        decode_responses=True,
                        socket_timeout=3,
                        socket_connect_timeout=3
                    )
                    docker_client.ping()
                    self.client = docker_client
                    self.redis_url = docker_host_url
                    self.is_connected = True
                    print(f"[RedisCache] Connected to Redis on host machine at {docker_host_url}")
                    self._load_semantic_index_from_redis()
                    return
                except Exception:
                    pass

            self.is_connected = False
            self.client = None
            print("[RedisCache] Operating with high-speed in-memory fallback cache.")

    def _load_semantic_index_from_redis(self):
        """Restores persisted semantic vector index from Redis."""
        global _SEMANTIC_INDEX
        if not self.is_connected or not self.client:
            return
        try:
            raw = self.client.get("dreamtrip:semantic:index")
            if raw:
                loaded = json.loads(raw)
                now = time.time()
                _SEMANTIC_INDEX = [item for item in loaded if item.get("expiry", 0) > now]
        except Exception as e:
            print(f"[RedisCache] Error loading semantic index: {e}")

    def _save_semantic_index_to_redis(self):
        """Persists semantic vector index to Redis."""
        if not self.is_connected or not self.client:
            return
        try:
            now = time.time()
            clean_index = [item for item in _SEMANTIC_INDEX if item.get("expiry", 0) > now]
            self.client.set("dreamtrip:semantic:index", json.dumps(clean_index), ex=86400)
        except Exception as e:
            print(f"[RedisCache] Error persisting semantic index: {e}")

    def _normalize_key(self, prefix: str, text: str) -> str:
        """Deterministic exact SHA-256 cache key."""
        cleaned = text.lower().strip()
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        hash_digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        return f"dreamtrip:{prefix}:{hash_digest}"

    def _canonical_key(self, prefix: str, intent: Dict[str, Any]) -> Optional[str]:
        """Generates a canonical intent cache key if destination & duration exist."""
        dest = str(intent.get("destination") or "").lower().strip()
        dest = re.sub(r"[^\w\s]", "", dest).replace(" ", "_")
        days = str(intent.get("duration_days") or "").strip()
        days_num = "".join(filter(str.isdigit, days)) or "default"
        budget = str(intent.get("budget_tier") or "moderate").lower().strip()

        if not dest or dest in ("none", "null"):
            return None

        return f"dreamtrip:{prefix}:canonical:{dest}_{days_num}d_{budget}"

    # --------------------------------------------------------------------------
    # Core Get & Set
    # --------------------------------------------------------------------------
    def get(self, prefix: str, text: str):
        """Standard exact retrieve."""
        key = self._normalize_key(prefix, text)
        now = time.time()

        if self.is_connected and self.client:
            try:
                raw_val = self.client.get(key)
                if raw_val is not None:
                    self.exact_hits += 1
                    try:
                        return json.loads(raw_val)
                    except json.JSONDecodeError:
                        return raw_val
            except Exception as e:
                print(f"[RedisCache] Redis GET error: {e}")

        if key in _MEMORY_CACHE:
            if now < _MEMORY_EXPIRY.get(key, 0):
                self.exact_hits += 1
                return _MEMORY_CACHE[key]
            else:
                del _MEMORY_CACHE[key]
                del _MEMORY_EXPIRY[key]

        return None

    def set(self, prefix: str, text: str, value, ttl_seconds: int = 86400) -> bool:
        """Standard exact store."""
        key = self._normalize_key(prefix, text)
        now = time.time()
        serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

        if self.is_connected and self.client:
            try:
                self.client.setex(key, ttl_seconds, serialized)
                return True
            except Exception as e:
                print(f"[RedisCache] Redis SET error: {e}")

        _MEMORY_CACHE[key] = value
        _MEMORY_EXPIRY[key] = now + ttl_seconds
        return True

    # --------------------------------------------------------------------------
    # Semantic Plan Caching Engine
    # --------------------------------------------------------------------------
    def get_cached_plan(self, query: str, canonical_intent: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], Optional[str]]:
        """
        Multi-Stage Semantic Retrieval:
        1. Exact SHA-256 match (< 1ms).
        2. Canonical Intent match (< 5ms) for identical destination/duration/budget.
        3. Semantic Vector Similarity match (Cosine similarity >= 0.88).

        Returns: (plan_data, cache_source_description) or (None, None).
        """
        now = time.time()

        # Step 1: Exact Hash Match
        exact_plan = self.get("plan", query)
        if exact_plan:
            return exact_plan, "Exact Cache (SHA-256)"

        # Step 2: Canonical Intent Match
        if canonical_intent:
            c_key = self._canonical_key("plan", canonical_intent)
            if c_key:
                # Check Redis
                if self.is_connected and self.client:
                    try:
                        raw = self.client.get(c_key)
                        if raw:
                            self.semantic_hits += 1
                            return json.loads(raw), "Canonical Intent Cache"
                    except Exception:
                        pass
                # Check Memory
                if c_key in _MEMORY_CACHE and now < _MEMORY_EXPIRY.get(c_key, 0):
                    self.semantic_hits += 1
                    return _MEMORY_CACHE[c_key], "Canonical Intent Cache"

        # Step 3: Semantic Vector Similarity Match
        if _SEMANTIC_INDEX:
            try:
                query_vector = get_embedding(query)
                best_similarity = 0.0
                best_entry = None

                for entry in _SEMANTIC_INDEX:
                    if entry.get("expiry", 0) < now:
                        continue
                    sim = cosine_similarity(query_vector, entry.get("vector", []))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_entry = entry

                # Threshold for semantic equivalence
                if best_similarity >= 0.88 and best_entry:
                    cached_key = best_entry["key"]
                    plan = None

                    if self.is_connected and self.client:
                        raw = self.client.get(cached_key)
                        if raw:
                            plan = json.loads(raw)
                    elif cached_key in _MEMORY_CACHE:
                        plan = _MEMORY_CACHE[cached_key]

                    if plan:
                        self.semantic_hits += 1
                        pct = round(best_similarity * 100)
                        return plan, f"Semantic Vector Cache ({pct}% Match)"

            except Exception as e:
                print(f"[RedisCache] Semantic matching error: {e}")

        # Cache Miss
        self.misses += 1
        return None, None

    def set_cached_plan(
        self,
        query: str,
        plan_data: dict,
        canonical_intent: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 86400
    ) -> bool:
        """
        Stores plan across:
        1. Exact SHA-256 key.
        2. Canonical Intent key (if intent available).
        3. Semantic Vector Index (for cosine similarity lookup).
        """
        global _SEMANTIC_INDEX
        now = time.time()
        expiry = now + ttl_seconds
        exact_key = self._normalize_key("plan", query)

        # 1. Save Exact Key
        self.set("plan", query, plan_data, ttl_seconds=ttl_seconds)

        # 2. Save Canonical Intent Key
        if canonical_intent:
            c_key = self._canonical_key("plan", canonical_intent)
            if c_key:
                serialized = json.dumps(plan_data)
                if self.is_connected and self.client:
                    try:
                        self.client.setex(c_key, ttl_seconds, serialized)
                    except Exception:
                        pass
                _MEMORY_CACHE[c_key] = plan_data
                _MEMORY_EXPIRY[c_key] = expiry

        # 3. Add to Semantic Vector Index
        try:
            vec = get_embedding(query)
            _SEMANTIC_INDEX = [item for item in _SEMANTIC_INDEX if item.get("expiry", 0) > now]
            _SEMANTIC_INDEX.append({
                "text": query,
                "vector": vec,
                "key": exact_key,
                "expiry": expiry
            })
            # Limit memory size to latest 500 semantic entries
            if len(_SEMANTIC_INDEX) > 500:
                _SEMANTIC_INDEX.pop(0)

            self._save_semantic_index_to_redis()
        except Exception as e:
            print(f"[RedisCache] Error indexing semantic vector: {e}")

        return True

    # --------------------------------------------------------------------------
    # Ancillary Tool Caches
    # --------------------------------------------------------------------------
    def get_cached_flight(self, query: str) -> Optional[str]:
        return self.get("flight", query)

    def set_cached_flight(self, query: str, flight_data: str, ttl_seconds: int = 43200) -> bool:
        return self.set("flight", query, flight_data, ttl_seconds=ttl_seconds)

    def get_cached_hotel(self, query: str) -> Optional[str]:
        return self.get("hotel", query)

    def set_cached_hotel(self, query: str, hotel_data: str, ttl_seconds: int = 86400) -> bool:
        return self.set("hotel", query, hotel_data, ttl_seconds=ttl_seconds)

    def record_token_savings(self, tokens: int):
        self.tokens_saved += tokens

    def clear_all(self) -> int:
        """Flushes Redis, memory caches, and semantic vector index."""
        global _SEMANTIC_INDEX
        cleared_count = 0
        if self.is_connected and self.client:
            try:
                keys = self.client.keys("dreamtrip:*")
                if keys:
                    cleared_count = self.client.delete(*keys)
            except Exception as e:
                print(f"[RedisCache] Clear error: {e}")

        cleared_count += len(_MEMORY_CACHE)
        _MEMORY_CACHE.clear()
        _MEMORY_EXPIRY.clear()
        _SEMANTIC_INDEX.clear()
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.tokens_saved = 0
        return cleared_count

    def get_stats(self) -> dict:
        total_requests = self.hits + self.misses
        hit_ratio = round((self.hits / total_requests) * 100, 1) if total_requests > 0 else 0.0
        return {
            "backend": "redis" if self.is_connected else "in_memory_fallback",
            "connected": self.is_connected,
            "redis_url": self.redis_url if self.is_connected else None,
            "total_hits": self.hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_ratio_percent": hit_ratio,
            "tokens_saved": self.tokens_saved,
            "semantic_index_size": len(_SEMANTIC_INDEX),
        }


# Global singleton cache instance for application-wide reuse
cache = RedisCacheManager()
