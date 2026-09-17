"""
================================================================================
DreamTrip AI — Redis Cache & Token Conservation Subsystem
================================================================================
This module provides enterprise-grade query and payload caching:
1. Normalizes user prompts (lowercasing, punctuation removal, whitespace collapse).
2. Generates deterministic SHA-256 keys: `dreamtrip:<prefix>:<hash>`.
3. Interacts with Redis (via SETEX with customized TTLs).
4. Provides an automatic, zero-dependency in-memory fallback if Redis is unreachable.
5. Tracks token savings analytics (cumulative tokens saved, cache hits, misses, hit ratio).
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. Standard Library & System Utilities
# ------------------------------------------------------------------------------
import os        # Operating system interface: reads REDIS_URL environment variable
import json      # Serializes complex dictionaries (plans, flights) to JSON strings for Redis
import hashlib   # Computes SHA-256 cryptographic digests for normalized cache keys
import time      # Tracks current epoch time for TTL enforcement in the in-memory fallback
import re        # Regular expressions for string cleansing and prompt normalization
from dotenv import load_dotenv  # Loads configuration from the .env file

# Ensure environment variables are loaded
load_dotenv()

# ------------------------------------------------------------------------------
# 2. Redis Client Import with Safe Fallback
# ------------------------------------------------------------------------------
try:
    import redis  # Official Redis Python driver supporting high-throughput key-value operations
except ImportError:
    redis = None

# Default Redis connection URL (local Redis server database 0)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# In-memory dictionary fallback structures used if the Redis server is unavailable
_MEMORY_CACHE = {}   # Stores payload: { normalized_key: value }
_MEMORY_EXPIRY = {}  # Stores expiry timestamp: { normalized_key: unix_timestamp }


class RedisCacheManager:
    """
    Manages caching lifecycle across travel plans, flight queries, and hotel searches.
    Includes automated failover to an in-memory dictionary if Redis is offline.
    """
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.client = None
        self.is_connected = False
        self.hits = 0          # Number of successful cache hits
        self.misses = 0        # Number of cache misses requiring fresh LLM/API execution
        self.tokens_saved = 0  # Estimated cumulative LLM tokens saved by cache hits
        self._init_connection()

    def _init_connection(self):
        """
        Initializes the Redis connection pool and validates connectivity with a PING.
        Falls back to in-memory caching gracefully if connection fails or redis is missing.
        """
        if not redis:
            print("[RedisCache] redis package not available. Using in-memory fallback.")
            return

        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,      # Automatically decodes Redis bytes to Python UTF-8 strings
                socket_timeout=3,           # Fail fast after 3 seconds if Redis server is unresponsive
                socket_connect_timeout=3
            )
            self.client.ping()              # Health check command to verify Redis server is alive
            self.is_connected = True
            print(f"[RedisCache] Successfully connected to Redis at {self.redis_url}")
        except Exception as e:
            self.is_connected = False
            self.client = None
            print(f"[RedisCache] Could not connect to Redis ({e}). Using in-memory fallback cache.")

    def _normalize_key(self, prefix: str, text: str) -> str:
        """
        Transforms user queries into normalized, deterministic cache keys.
        Example:
            "Plan a 7-day trip to Tokyo!" and "  plan a 7-day trip to tokyo  "
            both resolve to the EXACT same key: 'dreamtrip:plan:<sha256_hash>'.
        """
        cleaned = text.lower().strip()
        cleaned = re.sub(r"[^\w\s]", "", cleaned)  # Strip punctuation
        cleaned = re.sub(r"\s+", " ", cleaned)      # Collapse consecutive spaces into a single space
        hash_digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        return f"dreamtrip:{prefix}:{hash_digest}"

    def get(self, prefix: str, text: str):
        """
        Retrieves a cached value by prefix and query string.
        Checks Redis first; falls back to in-memory dictionary if disconnected.
        """
        key = self._normalize_key(prefix, text)
        now = time.time()

        # Check Redis if active
        if self.is_connected and self.client:
            try:
                raw_val = self.client.get(key)
                if raw_val is not None:
                    self.hits += 1
                    try:
                        return json.loads(raw_val)
                    except json.JSONDecodeError:
                        return raw_val
            except Exception as e:
                print(f"[RedisCache] Redis GET error: {e}")

        # Check In-Memory Fallback
        if key in _MEMORY_CACHE:
            if now < _MEMORY_EXPIRY.get(key, 0):
                self.hits += 1
                return _MEMORY_CACHE[key]
            else:
                # Key expired in fallback memory
                del _MEMORY_CACHE[key]
                del _MEMORY_EXPIRY[key]

        self.misses += 1
        return None

    def set(self, prefix: str, text: str, value, ttl_seconds: int = 86400) -> bool:
        """
        Stores a value with a Time-To-Live (TTL) expiration in seconds.
        Serializes dictionaries to JSON before saving to Redis.
        """
        key = self._normalize_key(prefix, text)
        now = time.time()
        serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

        # Store in Redis
        if self.is_connected and self.client:
            try:
                self.client.setex(key, ttl_seconds, serialized)
                return True
            except Exception as e:
                print(f"[RedisCache] Redis SET error: {e}")

        # Store in in-memory fallback
        _MEMORY_CACHE[key] = value
        _MEMORY_EXPIRY[key] = now + ttl_seconds
        return True

    # --------------------------------------------------------------------------
    # High-Level Domain Helpers
    # --------------------------------------------------------------------------
    def get_cached_plan(self, query: str) -> dict | None:
        """Retrieves cached master travel plan (TTL: 24h)."""
        return self.get("plan", query)

    def set_cached_plan(self, query: str, plan_data: dict, ttl_seconds: int = 86400) -> bool:
        """Caches synthesized travel master plan for 24 hours (86,400s)."""
        return self.set("plan", query, plan_data, ttl_seconds=ttl_seconds)

    def get_cached_flight(self, query: str) -> str | None:
        """Retrieves cached flight route schedules (TTL: 12h)."""
        return self.get("flight", query)

    def set_cached_flight(self, query: str, flight_data: str, ttl_seconds: int = 43200) -> bool:
        """Caches flight route schedules for 12 hours (43,200s)."""
        return self.set("flight", query, flight_data, ttl_seconds=ttl_seconds)

    def get_cached_hotel(self, query: str) -> str | None:
        """Retrieves cached hotel stay options (TTL: 24h)."""
        return self.get("hotel", query)

    def set_cached_hotel(self, query: str, hotel_data: str, ttl_seconds: int = 86400) -> bool:
        """Caches hotel stay options for 24 hours (86,400s)."""
        return self.set("hotel", query, hotel_data, ttl_seconds=ttl_seconds)

    def record_token_savings(self, tokens: int):
        """Accumulates calculated token savings from cache hits."""
        self.tokens_saved += tokens

    def clear_all(self) -> int:
        """
        Deletes all 'dreamtrip:*' keys in Redis and clears in-memory fallback storage.
        Resets hit/miss counters and tokens saved.
        """
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
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        return cleared_count

    def get_stats(self) -> dict:
        """
        Compiles performance metrics for the frontend Telemetry dashboard:
        - Connection status and backend type
        - Hits, misses, total requests, hit ratio %
        - Cumulative tokens conserved
        """
        total_requests = self.hits + self.misses
        hit_ratio = round((self.hits / total_requests) * 100, 1) if total_requests > 0 else 0.0
        return {
            "backend": "redis" if self.is_connected else "in_memory_fallback",
            "connected": self.is_connected,
            "redis_url": self.redis_url if self.is_connected else None,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_ratio_percent": hit_ratio,
            "tokens_saved": self.tokens_saved,
        }


# Global singleton cache instance for application-wide reuse
cache = RedisCacheManager()
