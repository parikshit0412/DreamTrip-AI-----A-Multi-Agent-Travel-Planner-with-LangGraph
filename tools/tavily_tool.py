"""
================================================================================
DreamTrip AI — Hotel & Stays Discovery Subsystem (Tavily AI Search Engine)
================================================================================
This module powers the Hotel Agent in the LangGraph multi-agent architecture:
1. Interfaces with Tavily's search engine, built specifically for AI agents.
2. Retrieves up-to-date accommodations, boutique stays, hostels, and prices.
3. Truncates content snippets to 300 characters to prevent LLM context overflow.
4. Caches hotel results in Redis with a 24-hour TTL (86,400s) to conserve tokens.
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. External APIs & Environment Configuration
# ------------------------------------------------------------------------------
import os                        # Operating system interface: reads TAVILY_API_KEY
from dotenv import load_dotenv  # Loads API keys from the local .env configuration file
from tavily import TavilyClient # Official client library for Tavily AI search engine

# Load environment variables
load_dotenv()

# Instantiate Tavily API client with credentials
client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# ------------------------------------------------------------------------------
# 2. Redis Cache Integration
# ------------------------------------------------------------------------------
from tools.redis_cache import cache  # High-performance Redis caching singleton (24h TTL)


# ==============================================================================
# Public API Function: Search Hotels with Redis Caching
# ==============================================================================
def tavily_search(query: str) -> str:
    """
    Executes an AI-powered web search for accommodations and hotels:
    1. Checks Redis cache for previously discovered hotels for this destination.
    2. Queries Tavily API (max 5 high-relevance results).
    3. Normalizes and trims snippets to avoid token blowout.
    4. Caches the compiled string in Redis with 24h TTL before returning.
    """
    # 1. Check Redis Cache first
    cached_val = cache.get_cached_hotel(query)
    if cached_val:
        return cached_val

    # 2. Query Tavily Search API
    try:
        response = client.search(query=query, max_results=5)
    except Exception as e:
        return f"Hotel search unavailable: {e}"

    # 3. Extract and sanitize results
    result = []
    for i, r in enumerate(response.get("results", [])):
        title = r.get("title", "No title")
        url = r.get("url")
        snippet = r.get("content", "").strip()

        # Keep only the first 300 characters of each snippet to conserve LLM tokens
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        result.append(
            f"[{i+1} Title:{title}]\nURL:{url}\n{snippet}"
        )

    # 4. Join and cache in Redis with 24-Hour TTL
    formatted = "\n\n".join(result)
    cache.set_cached_hotel(query, formatted, ttl_seconds=86400)
    return formatted