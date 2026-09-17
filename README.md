# ✈️ DreamTrip AI — Autonomous Multi-Agent Travel Planner

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.2-FF6F00.svg?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Redis](https://img.shields.io/badge/Redis-8.1%20%2F%207%20Alpine-DC382D.svg?logo=redis&logoColor=white)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PostgresSaver-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**An enterprise-grade autonomous travel orchestration engine powered by LangGraph cyclical StateGraphs, Google Gemini 2.5 Flash, PostgreSQL persistent checkpointing, and a high-performance Redis token conservation cache.**

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Token Economy & Benchmarks](#-token-economy--benchmarks) • [Quickstart Guide](#-quickstart-guide) • [Docker Deployment](#option-a-docker-compose-fastest--zero-setup) • [API Reference](#-api-reference)

</div>

---

## 📖 Executive Summary

**DreamTrip AI** transforms natural language travel requests into comprehensive, logistics-aware master itineraries. Traditional travel chatbots often hallucinate flight schedules, recommend closed hotels, or make redundant, costly LLM calls for identical queries. 

DreamTrip AI overcomes these limitations with an **autonomous multi-agent architecture**:
1. **Parallel Tool Specialists:** An Aviation Agent (AviationStack API + offline IATA database) and a Stays Agent (Tavily AI Search) execute simultaneously in parallel.
2. **Deterministic Synthesis:** An Itinerary Agent and an Executive Synthesis Agent compile verified schedules into realistic, budget-conscious day-by-day itineraries.
3. **Sub-10ms Redis Token Caching:** Normalizes queries and caches plans, saving **100% of LLM tokens** on repeat or semantically identical requests.
4. **State Checkpointing:** Backed by `PostgresSaver` for thread-level state recovery and conversational history.
5. **Executive PDF Export:** High-contrast, publication-grade document generation designed for travelers on the go.

---

## ⚡ Key Features

- **Multi-Agent Parallel Orchestration:** Built with **LangGraph StateGraph**. Concurrently queries flight routes and hotel availability via directed acyclic fan-out nodes, slashing orchestration latency.
- **Sub-10ms Redis Token Conservation Engine:** 
  - Canonical prompt normalization (casing, punctuation, whitespace sanitization).
  - Deterministic SHA-256 key hashing (`dreamtrip:plan:<hash>`).
  - Tiered TTLs: **24 hours** for full itineraries and hotel suggestions; **12 hours** for real-time flight schedules.
  - Zero-configuration **in-memory fallback** if Redis is temporarily unavailable.
- **Zero-Quota Offline Airport Resolution:** Integrates `airportsdata` (40,000+ global airfields) and `pycountry` to geocode cities into 3-letter IATA codes (e.g. "Tokyo" &rarr; `NRT`/`HND`) locally without consuming third-party API quotas.
- **Enterprise State Persistence:** Implements PostgreSQL `PostgresSaver` checkpointing, preserving multi-turn chat sessions across restarts.
- **Real-Time Telemetry Dashboard:** Live frontend telemetry tracking cache hits, cache misses, hit ratio percentage, and cumulative tokens conserved.
- **Luxury Single-Page Application (SPA):** Modern crimson & amber theme (`#991b1b` / `#f59e0b`), glassmorphic panels, dynamic prompt chips, and responsive layout.
- **Executive PDF Export Engine:** One-click PDF generation via `html2pdf.js` with dedicated print styling, high-contrast typography, and zero canvas clipping.
- **Production-Ready Containerization:** Multi-stage `Dockerfile` and `docker-compose.yml` orchestrating the FastAPI application alongside a healthy Redis 7 Alpine service.

---

## 🏗️ System Architecture

The following diagram illustrates the lifecycle of a travel query through the caching layer, LangGraph agent network, and persistent storage:

```mermaid
flowchart TD
    User([👤 User / Client SPA]) -->|POST /api/travel| API[FastAPI Gateway]
    
    %% Redis Cache Interception Layer
    API -->|1. Check Cache| CacheCheck{Redis Cache HIT?}
    CacheCheck -- YES (< 10ms) -->|Return Cached Master Plan| CachedResp[⚡ 0 LLM Calls | ~6,000 Tokens Conserved]
    CachedResp -->|Instant JSON| User
    
    %% Multi-Agent LangGraph Workflow
    CacheCheck -- NO (Cache Miss) --> GraphInit[🚀 Initialize LangGraph StateGraph]
    
    subgraph AgentPipeline ["Multi-Agent Execution Graph (LangGraph)"]
        direction TB
        STARTNode([START]) --> FlightAgent["✈️ Flight Intelligence Agent<br/>(AviationStack API + IATA Engine)"]
        STARTNode --> HotelAgent["🏨 Hotel Discovery Agent<br/>(Tavily AI Search)"]
        
        FlightAgent --> ItineraryAgent["📅 Sequential Itinerary Agent<br/>(Gemini 2.5 Flash)"]
        HotelAgent --> ItineraryAgent
        
        ItineraryAgent --> FinalAgent["📝 Executive Synthesis Agent<br/>(Gemini 2.5 Flash)"]
        FinalAgent --> ENDNode([END])
    end
    
    GraphInit --> AgentPipeline
    
    %% Persistence and Storage
    AgentPipeline -.->|Checkpoints & Thread State| Postgres[(🐘 PostgreSQL / PostgresSaver)]
    AgentPipeline -->|2. Store Master Plan 24h TTL| RedisSet[(⚡ Redis 8 / Docker Redis)]
    
    FinalAgent -->|3. Compile Response| APIResponse[HTTP 200 JSON Response]
    APIResponse -->|Formatted Itinerary + Telemetry| User
```

### Agent Node Breakdown

| Agent Node | Technology / Tool | Responsibility |
|:---|:---|:---|
| **Flight Intelligence Agent** | `AviationStack API` + `airportsdata` | Extracts origin/destination, resolves IATA airport codes offline, and fetches real-time departure/arrival schedules. |
| **Hotel Discovery Agent** | `Tavily AI Search` | Discovers curated accommodations, luxury resorts, boutique stays, and price-to-quality ratios. |
| **Itinerary Agent** | `Google Gemini 2.5 Flash` | Merges flight logistics and hotel locations into a realistic, chronologically sound day-by-day itinerary. |
| **Executive Synthesis Agent** | `Google Gemini 2.5 Flash` | Formats the final output into 6 structured sections: Trip Summary, Flights, Hotels, Itinerary, Budget, and Recommendations. |

---

## 📊 Token Economy & Benchmarks

Repeated or overlapping travel queries are a major source of latency and LLM cost. DreamTrip AI's Redis caching layer eliminates redundant compute:

| Metric | Fresh Generation (Cache Miss) | Redis Cached (Cache Hit) | Improvement |
|:---|:---:|:---:|:---:|
| **Response Latency** | `67.42 seconds` | **`4.1 milliseconds`** | **⚡ 99.99% Faster** |
| **LLM Model Calls** | `4 calls` | **`0 calls`** | **💯 100% Elimination** |
| **Tokens Consumed** | `~6,059 tokens` | **`0 tokens`** | **📉 100% Reduction** |
| **Tokens Saved Recorded** | `0 tokens` | **`+6,059 tokens`** | **💰 Immediate Cost Savings** |
| **External API Quota** | 2 API requests (Flights + Hotels) | **0 API requests** | **🛡️ Zero Quota Depletion** |

> **Live Verification:** You can inspect real-time metrics anytime at the `/api/cache/stats` endpoint or directly in the application's header telemetry badge.

---

## 📁 Repository Structure

```text
DreamTrip-AI-----A-Multi-Agent-Travel-Planner-with-LangGraph/
├── app.py                     # FastAPI web server, route handlers & thread pool manager
├── backend.py                 # LangGraph StateGraph, PostgreSQL checkpointer & orchestrator
├── Dockerfile                 # Multi-stage production container definition
├── docker-compose.yml         # Container orchestration (Web App + Redis 7 Alpine)
├── .dockerignore              # Excludes virtualenvs, cache, and secrets from image
├── .env.example               # Template environment configuration file
├── requirements.txt           # Pinned production Python dependencies
├── test.py                    # Standalone CLI test suite for graph execution
├── LICENSE                    # MIT Open Source License
├── tools/
│   ├── __init__.py            # Python package initialization
│   ├── flight_tool.py         # AviationStack client, IATA geocoder & flight cache
│   ├── redis_cache.py         # Redis Cache Manager, key normalizer & in-memory fallback
│   └── tavily_tool.py         # Tavily Search API client & hotel cache
├── templates/
│   └── index.html             # Luxury single-page application (Jinja2 / HTML5)
└── static/
    ├── style.css              # Crimson & Gold design system, glassmorphism & responsive CSS
    └── script.js              # State manager, async polling, PDF generation & telemetry
```

---

## 🚀 Quickstart Guide

You can run DreamTrip AI either via **Docker Compose (Recommended)** or directly in a **local Python virtual environment**.

### Prerequisites

Ensure you have obtained API keys for:
1. **[Google AI Studio](https://aistudio.google.com/)**: For `GEMINI_API_KEY` (Gemini 2.5 Flash).
2. **[Tavily AI](https://tavily.com/)**: For `TAVILY_API_KEY` (Web search & hotel discovery).
3. **[AviationStack](https://aviationstack.com/)**: For `AVIATIONSTACK_API_KEY` (Live flight schedules).
4. **PostgreSQL Database**: Free cloud instance from [Neon](https://neon.tech/), [Supabase](https://supabase.com/), [Render](https://render.com/), or local PostgreSQL.

---

### Option A: Docker Compose (Fastest & Zero Setup)

Ensure [Docker](https://www.docker.com/) and Docker Compose are installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parikshit0412/DreamTrip-AI-----A-Multi-Agent-Travel-Planner-with-LangGraph.git
   cd DreamTrip-AI-----A-Multi-Agent-Travel-Planner-with-LangGraph
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` in your editor and enter your credentials.

3. **Start the application and Redis:**
   ```bash
   docker compose up --build -d
   ```

4. **Access the application:**
   Open your browser and navigate to:
   ```text
   http://localhost:8000
   ```

To stop containers:
```bash
docker compose down
```

---

### Option B: Local Virtual Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parikshit0412/DreamTrip-AI-----A-Multi-Agent-Travel-Planner-with-LangGraph.git
   cd DreamTrip-AI-----A-Multi-Agent-Travel-Planner-with-LangGraph
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure your `.env` file:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   DATABASE_URL=postgresql://user:password@host:5432/dbname?sslmode=require
   AVIATIONSTACK_API_KEY=your_aviationstack_api_key
   TAVILY_API_KEY=your_tavily_api_key
   REDIS_URL=redis://localhost:6379/0
   DEFAULT_ORIGIN_IATA=DAC
   ```

   *(Note: If you do not have a local Redis server running, DreamTrip AI automatically falls back to its built-in in-memory caching engine without throwing errors).*

5. **Start the FastAPI server:**
   ```bash
   python app.py
   ```
   Or via Uvicorn directly:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

6. **Open in browser:**
   ```text
   http://127.0.0.1:8000
   ```

---

## ⚙️ Environment Variables Reference

| Variable | Required | Default | Description |
|:---|:---:|:---:|:---|
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini API key powering the reasoning and synthesis agents. |
| `DATABASE_URL` | **Yes** | — | PostgreSQL connection string for LangGraph `PostgresSaver` checkpointing. |
| `AVIATIONSTACK_API_KEY` | **Yes** | — | AviationStack REST API key for live flight discovery. |
| `TAVILY_API_KEY` | **Yes** | — | Tavily Search API key for curated hotel and stay intelligence. |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Connection URL for Redis query and token caching. |
| `DEFAULT_ORIGIN_IATA` | No | `DAC` | Default departure airport IATA code when none is stated (e.g. `JFK`, `LHR`, `DAC`). |
| `PORT` | No | `8000` | Application port used by Docker containers. |

---

## 🔌 API Reference

### 1. Generate Travel Itinerary
- **Endpoint:** `POST /api/travel`
- **Content-Type:** `application/json`

**Request Body:**
```json
{
  "message": "Plan a 7-day trip to Tokyo from New York with a $3000 budget",
  "thread_id": "user_session_abc123"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "thread_id": "user_session_abc123",
  "answer": "### 1. Trip Summary\n...",
  "flight_results": "Flight AA167 departing JFK...",
  "hotel_results": "1. Hotel Gracery Shinjuku...",
  "itinerary": "Day 1: Arrival & Shibuya Crossing...",
  "llm_calls": 0,
  "is_cached": true,
  "cache_source": "Redis",
  "tokens_saved": 6059
}
```

---

### 2. Cache Telemetry Stats
- **Endpoint:** `GET /api/cache/stats`

**Response (200 OK):**
```json
{
  "backend": "redis",
  "connected": true,
  "redis_url": "redis://localhost:6379/0",
  "hits": 14,
  "misses": 3,
  "total_requests": 17,
  "hit_ratio_percent": 82.4,
  "tokens_saved": 84826
}
```

---

### 3. Flush Cache
- **Endpoint:** `POST /api/cache/clear`

**Response (200 OK):**
```json
{
  "success": true,
  "cleared_keys": 8,
  "message": "Cache flushed successfully."
}
```

---

### 4. Health Check
- **Endpoint:** `GET /health`

**Response (200 OK):**
```json
{
  "status": "ok",
  "message": "AI Travel Planner API is running",
  "redis_connected": true
}
```

---

## 📄 PDF Export Engine

DreamTrip AI includes a custom-engineered client-side PDF export workflow:
- **Foreground Staging Modal (`#pdfStagingOverlay`):** Renders off-screen content into a dedicated, clean, high-contrast DOM node before rasterization.
- **Color Contrast & Typography:** Deep slate typography (`#0f172a`), luxury crimson headers (`#881337`), and golden borders (`#b45309`) designed specifically for printed physical paper or digital tablets.
- **Smart Pagination:** Optimized CSS page-break rules (`break-inside: avoid-page`) prevent orphan headers and eliminate blank trailing pages.

---

## 🧪 CLI Testing

To verify agent execution and tool connectivity directly from the terminal:

```bash
python test.py
```

Enter a prompt when prompted (e.g. `Plan a 5-day trip to Paris from London`). The script invokes the LangGraph pipeline and prints the final itinerary to standard output.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

Built with ❤️ by [Parikshit](https://github.com/parikshit0412) using **LangGraph**, **Google Gemini**, and **FastAPI**.

</div>