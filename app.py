"""
================================================================================
DreamTrip AI — FastAPI Web Application & API Gateway
================================================================================
This module serves as the production web application gateway:
1. Mounts static asset directories (CSS, JS, media).
2. Serves the interactive Jinja2 HTML templates.
3. Exposes the asynchronous `/api/travel` orchestration endpoint.
4. Exposes Redis cache telemetry (`/api/cache/stats`) and management (`/api/cache/clear`).
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. Standard Library & System Utilities
# ------------------------------------------------------------------------------
from pathlib import Path       # Object-oriented filesystem paths for safe directory resolution
import traceback               # Formats detailed exception stack traces for server debugging
import uvicorn                 # Lightning-fast ASGI web server implementation for FastAPI
import warnings                # Controls runtime warning filters

# Suppress benign third-party library warnings
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*google.*")

# ------------------------------------------------------------------------------
# 2. FastAPI & Web Framework Core
# ------------------------------------------------------------------------------
from fastapi import FastAPI, Request                 # Core framework and incoming HTTP request context
from fastapi.responses import HTMLResponse, JSONResponse # Explicit HTTP response type handlers
from fastapi.staticfiles import StaticFiles          # Static file mounting handler (for CSS/JS/images)
from fastapi.templating import Jinja2Templates        # Template engine for rendering HTML
from pydantic import BaseModel                       # Data validation library ensuring strongly-typed request schemas
from starlette.concurrency import run_in_threadpool  # Offloads synchronous LangGraph graph execution to a thread pool

# ------------------------------------------------------------------------------
# 3. Application Domain Imports
# ------------------------------------------------------------------------------
from backend import run_travel_agent  # Multi-agent LangGraph orchestrator
from tools.redis_cache import cache   # Singleton Redis cache manager for token savings & telemetry

# Resolve base project directory
BASE_DIR = Path(__file__).resolve().parent

# Instantiate FastAPI application
app = FastAPI(
    title="DreamTrip AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend & Redis Token Conservation",
    version="1.0.0"
)

# Mount static files directory (/static -> d:\...\static)
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

# Configure Jinja2 templates directory (/templates -> d:\...\templates)
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# ==============================================================================
# Pydantic Request Validation Models
# ==============================================================================
class TravelRequest(BaseModel):
    """
    Schema for incoming travel planning requests from the client.
    Validates message existence and optionally captures an existing thread_id for conversation state.
    """
    message: str                 # The user's travel query (destination, budget, preferences)
    thread_id: str | None = None # Optional session ID tracked by the PostgreSQL checkpointer


# ==============================================================================
# Web & API Route Handlers
# ==============================================================================

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the single-page application (SPA) user interface.
    Handles HEAD requests cleanly for cloud health probes (e.g. Render, Cloudflare).
    """
    if request.method == "HEAD":
        return HTMLResponse(content="", status_code=200)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    """
    Primary API endpoint for trip synthesis:
    - Checks Redis cache (instant response, 0 tokens consumed).
    - On cache miss, executes the multi-agent LangGraph pipeline in a background thread pool.
    - Saves result to Redis for future identical/normalized prompts.
    """
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        # Execute LangGraph in thread pool to prevent blocking the async FastAPI event loop
        result = await run_in_threadpool(
            run_travel_agent,
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
                "llm_calls": result["llm_calls"],
                "is_cached": result.get("is_cached", False),
                "cache_source": result.get("cache_source"),
                "tokens_saved": result.get("tokens_saved", 0),
            }
        )

    except Exception as e:
        print("ERROR during travel planner execution:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/api/cache/stats")
async def cache_stats():
    """
    Exposes live telemetry from the Redis cache manager:
    Returns hits, misses, hit ratio %, and cumulative tokens saved.
    """
    return cache.get_stats()


@app.post("/api/cache/clear")
async def cache_clear():
    """
    Flushes all cached travel plans, flights, and hotel records from Redis.
    Useful for demonstrating fresh LLM generation vs. cached retrieval in interviews.
    """
    count = cache.clear_all()
    return {"success": True, "cleared_keys": count, "message": "Cache flushed successfully."}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    """
    System health check reporting service status and Redis connectivity.
    Supports both GET and HEAD probes.
    """
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running",
        "redis_connected": cache.is_connected
    }


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
async def favicon():
    """
    Handles browser favicon requests cleanly to prevent unnecessary 404 error logs.
    """
    return JSONResponse(content={})


# ==============================================================================
# Local Server Execution Entrypoint
# ==============================================================================
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Auto-reloads server upon code changes during development
    )