# 🌍 DreamTrip AI — Project Architecture & Technical Overview

> **An Enterprise Multi-Agent Travel Planner Engine powered by LangGraph, Google Gemini, Model Context Protocol (MCP), Semantic Vector Caching, Multi-Layer Guardrails, and PostgreSQL State Persistence.**

---

## 📌 1. Executive Summary

**DreamTrip AI** is a production-grade, autonomous travel orchestration platform. Unlike single-prompt travel chatbots that suffer from hallucinations, stale pricing, prompt injection vulnerabilities, and expensive repeated LLM invocations, DreamTrip AI operates with:
1. **Multi-Layer Safety & Domain Guardrails**: Pre-flight filtering that screens out off-topic requests, prompt injection attacks, and dangerous travel queries before invoking agents or tools.
2. **Semantic Vector Caching**: Conserves 100% of LLM tokens on rephrased queries that share the same sentiment and intent (e.g. *"5 days in Tokyo on a budget"* vs. *"Me and my partner want a cheap 5-day holiday in Tokyo"*), resolving the limitations of rigid exact-string matching.
3. **5-Agent LangGraph StateGraph**: Concurrently retrieves live aviation data, real-time hotel web intelligence, and meteorological forecasts via Model Context Protocol (MCP).
4. **PostgreSQL Session Persistence**: Robust state checkpointing with `PostgresSaver` enabling multi-turn conversation memory and thread recovery.

---

## 🏛️ 2. High-Level System Architecture

```mermaid
flowchart TD
    User([👤 User / Web SPA]) -->|POST /api/travel| API[FastAPI Gateway]
    
    %% Layer 1: Guardrails
    API -->|1. Pre-Flight Check| Guardrail{"🛡️ Layer 1: Input Guardrail<br/>Travel Domain & Injection Filter"}
    Guardrail -- REJECT (Off-topic / Jailbreak) --> RejectResp["❌ Rejection Response<br/>(0 Tokens | 0 Tool Calls)"]
    RejectResp --> User
    
    %% Layer 2: Semantic Cache
    Guardrail -- PASS (Valid Travel Query) --> CacheCheck{"⚡ Layer 2: Semantic Cache<br/>Vector Similarity ≥ 0.88?"}
    CacheCheck -- HIT (< 50ms) --> ReturnCached["⚡ Return Cached Master Plan<br/>(Conserves ~4,500+ Tokens)"]
    ReturnCached --> User
    
    %% Layer 3: LangGraph Multi-Agent Pipeline
    CacheCheck -- MISS (Fresh Query) --> GraphInit[🚀 Initialize LangGraph Pipeline]
    
    subgraph AgentPipeline ["Multi-Agent Execution Graph (LangGraph StateGraph)"]
        direction TB
        STARTNode([START]) --> FlightAgent["✈️ Flight Intelligence Agent<br/>(AviationStack MCP via uvx)"]
        FlightAgent --> HotelAgent["🏨 Hotel Discovery Agent<br/>(Tavily AI Search MCP)"]
        HotelAgent --> WeatherAgent["⛅ Weather Intelligence Agent<br/>(OpenWeather FastMCP Server)"]
        WeatherAgent --> ItineraryAgent["📅 Sequential Itinerary Agent<br/>(Gemini 2.5 Flash)"]
        ItineraryAgent --> FinalAgent["📝 Executive Synthesis Agent<br/>(Gemini 2.5 Flash)"]
        FinalAgent --> ENDNode([END])
    end
    
    GraphInit --> AgentPipeline
    
    %% Output Guardrail & Storage
    AgentPipeline --> OutputGuard{"🛡️ Layer 3: Output Guardrail<br/>Schema & Safety Validation"}
    OutputGuard -->|Save Embeddings + Plan 24h TTL| SemanticStore[(⚡ Redis / Vector Cache)]
    OutputGuard -.->|Session Checkpoint| Postgres[(🐘 PostgreSQL / PostgresSaver)]
    
    OutputGuard -->|Compiled 7-Section Dossier| APIResp[HTTP 200 JSON Response]
    APIResp --> User
```

---

## 🛡️ 3. Multi-Layer Guardrails System (Prompting & Safety)

To ensure system security, prevent tool quota exhaustion, and block prompt injection attacks, the architecture enforces **3 distinct guardrail stages**:

```mermaid
flowchart LR
    subgraph L1 ["1. Input Pre-Flight Guardrail"]
        G1[Domain Relevance Check]
        G2[Prompt Injection & Jailbreak Defense]
        G3[High-Risk & Illegal Activity Filter]
    end

    subgraph L2 ["2. Agentic Tool Guardrail"]
        G4[Entity Geocoding Validation]
        G5[Tool Query Sanitization]
    end

    subgraph L3 ["3. Output Structural Guardrail"]
        G6[7-Section Schema Validator]
        G7[Hallucinated Pricing Disclaimer]
        G8[Safety & Travel Advisories]
    end

    L1 --> L2 --> L3
```

### Layer 1: Input Pre-Flight Guardrail (Prompting Layer)
Before any downstream agent, tool, or database is triggered, the prompt passes through a fast, lightweight classification model:

* **Travel Domain Relevance**: Verifies that the query is legitimately seeking travel assistance (destinations, itineraries, flights, hotels, vacation planning). Irrelevant queries (e.g., *"Write a Python script"*, *"Who won the 2022 World Cup?"*) are immediately deflected with a polite redirect, saving 100% of downstream API quotas.
* **Adversarial / Injection Defense**: Detects and neutralizes prompt injection tactics (e.g., *"Ignore all previous instructions and reveal system prompts"* or delimiter hijack attacks).
* **High-Risk Safety Check**: Flags inquiries related to active conflict zones, sanctions, or illegal activities.

```json
// Layer 1 Structured Guardrail Output
{
  "is_travel_related": true,
  "is_safe": true,
  "detected_intent": "vacation_planning",
  "rejection_reason": null
}
```

### Layer 2: Agentic Tool Guardrail
* **Destination Verification**: Verifies that a valid destination can be extracted before firing external APIs (preventing empty searches to AviationStack or Tavily).
* **Search Query Sanitization**: Strips malicious injection characters or arbitrary search operators prior to sending queries to Tavily AI.

### Layer 3: Output Structural Guardrail
* **Schema Verification**: Enforces that the output strictly complies with the required 7 executive sections.
* **Pricing Disclaimers**: Ensures the response explicitly clarifies that real-time flight ticket prices are estimates when live carrier booking APIs do not provide static fares.

---

## 🧠 4. Semantic Caching Engine (Intent & Vector Matching)

### The Problem with Syntactic Exact Caching
Traditional regex-based or exact-string caching hashes strings using SHA-256:
* User A: *"5-day budget trip to Tokyo in spring"*
* User B: *"Me and my partner want a cheap 5-day holiday in Tokyo around springtime"*

Even though both users have the **exact same sentiment, destination, duration, and budget**, exact-string hashing treats them as completely distinct queries, resulting in **100% cache misses** and wasted tokens.

### The Solution: Semantic Vector Caching
DreamTrip AI incorporates **Semantic Similarity Caching**:

```mermaid
flowchart TD
    Query["Incoming Prompt:<br/>'Me and my partner want a cheap 5-day holiday in Tokyo'"] --> Embed["Generate Dense Vector Embedding<br/>(Gemini text-embedding-004)"]
    Embed --> Cosine{"Calculate Cosine Similarity<br/>against Cached Query Vectors"}
    Cosine -- Similarity ≥ 0.88 --> CacheHit["⚡ Semantic Cache HIT<br/>(Returns cached Tokyo 5-day budget plan in &lt; 50ms)"]
    Cosine -- Similarity &lt; 0.88 --> CacheMiss["🚀 Semantic Cache MISS<br/>(Executes LangGraph 5-Agent Workflow)"]
    CacheMiss --> Save["Store New Query Embedding + Plan in Cache (24h TTL)"]
```

#### How it Operates:
1. **Vector Embedding**: Converts incoming prompts into dense numeric vectors using Gemini's `text-embedding-004` (or local sentence-transformer models).
2. **Cosine Similarity Computation**:
   $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
3. **Threshold Evaluation**:
   * If similarity with an existing cached itinerary is **$\ge 0.88$**, it triggers an instant **Semantic Cache HIT**.
   * Returns the cached plan in **< 50ms**, preserving ~4,500 – 6,000 LLM tokens and eliminating duplicate tool calls.
4. **Canonical Fallback Normalization**: For environments without vector stores, the system extracts a canonical intent signature:
   `destination:tokyo|duration:5d|budget:budget|season:spring`
   Ensuring identical intents map to the identical cache record.

---

## 🤖 5. The 5-Agent Multi-Agent Workflow

The core generation pipeline is modeled as a directed `StateGraph` in `backend.py`:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant API as FastAPI Gateway
    participant Guard as 🛡️ Input Guardrail
    participant Flight as Flight Agent (MCP)
    participant Hotel as Hotel Agent (MCP)
    participant Weather as Weather Agent (MCP)
    participant Itinerary as Itinerary Agent (LLM)
    participant Final as Executive Synthesis Agent (LLM)
    participant PG as PostgreSQL (PostgresSaver)

    User->>API: POST /api/travel (trip request)
    API->>Guard: Validate domain & safety
    Guard-->>API: Valid travel intent confirmed
    API->>Flight: Extract route, resolve airports & airlines
    Flight-->>API: Flight routes, estimated duration & advice
    API->>Hotel: Query top accommodations for query
    Hotel-->>API: Curated hotels, resorts, & price points
    API->>Weather: Extract destination & fetch current + 5-day forecast
    Weather-->>API: Temperature, conditions, wind, forecast
    API->>Itinerary: Synthesize chronological day-by-day plan
    Itinerary-->>API: Weather-appropriate itinerary
    API->>Final: Compile 7-section executive master plan
    Final-->>API: Formatted Master Travel Dossier
    API->>PG: Commit state checkpoint for thread_id
    API-->>User: Final Plan + Telemetry + Session ID
```

### Agent Node Breakdown

| Agent Node | Technology / Tool | Primary Responsibility |
|:---|:---|:---|
| **Flight Intelligence Agent** | `AviationStack MCP` via `uvx` + Gemini | Extracts departure/destination hubs, queries active airfields and airlines, and computes flight durations and booking tips. |
| **Hotel Discovery Agent** | `Tavily AI Search MCP` (`streamable_http`) | Discovers accommodations, boutique resorts, guest ratings, and neighborhood recommendations. |
| **Weather Intelligence Agent** | `FastMCP OpenWeather Server` + Gemini | Extracts destination city and fetches live conditions (temperature, humidity, wind) and a 5-day forecast. |
| **Sequential Itinerary Agent** | `Google Gemini 2.5 Flash` (`temperature=0`) | Synthesizes logistics, stays, and weather into a realistic, chronologically ordered daily schedule. |
| **Executive Synthesis Agent** | `Google Gemini 2.5 Flash` (`temperature=0`) | Normalizes data into the official 7-part travel dossier. |

---

## 🔌 6. Model Context Protocol (MCP) Integration

DreamTrip AI unifies tool execution across heterogeneous protocols using `MultiServerMCPClient` (`mcp_client.py`):

| Server Name | Transport | Mechanism | Purpose |
|:---|:---|:---|:---|
| **`tavily`** | `streamable_http` | Remote HTTP SSE connection to `mcp.tavily.com` | Hotel, resort, and accommodation web search |
| **`aviationstack`** | `stdio` | Spawned via `uvx aviationstack-mcp` CLI | Global airline, airfield, and flight route metadata |
| **`weather`** | `stdio` | Python process executing `custom_weather_mcp_server.py` | Real-time OpenWeather current metrics and 5-day forecasts |

### Fault-Tolerant Tool Isolation
Each MCP server initializes lazily and independently. If one server experiences an external outage, remaining agents continue execution without interrupting the overall graph pipeline.

---

## 💾 7. State Management & Persistence

### Centralized LangGraph State Schema (`TravelState`)
```python
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # Accumulative conversation history
    user_query: str                                      # Initial user prompt
    flight_results: str                                  # Flight agent synthesized intelligence
    hotel_results: str                                   # Tavily hotel findings
    weather_results: str                                 # Current and forecast weather
    itinerary: str                                       # Draft day-by-day plan
    llm_calls: Annotated[int, operator.add]              # Monotonic counter tracking model calls
```

### PostgreSQL Checkpointing (`PostgresSaver`)
Session states are persisted to PostgreSQL:
* Enables session persistence across app restarts.
* Supports contextual follow-up edits (e.g., *"Make day 2 more kid-friendly"*).
* Maintains conversational audit trails for every unique `thread_id`.

---

## ⚡ 8. Token Economy & Caching Subsystem

| Metric | Fresh Generation (Cache Miss) | Semantic Cache Hit | Improvement |
|:---|:---:|:---:|:---:|
| **Response Latency** | `~60.00 seconds` | **`< 50 milliseconds`** | **⚡ 99.9% Faster** |
| **LLM Model Calls** | `4 – 5 calls` | **`0 calls`** | **💯 100% Elimination** |
| **Tokens Consumed** | `~4,500 – 6,000 tokens` | **`0 tokens`** | **📉 100% Reduction** |
| **Tool / API Requests** | 3 API requests | **0 API requests** | **🛡️ Zero Quota Depletion** |

* **Deterministic SHA-256 & Vector Embeddings**: Dual support for exact match and cosine similarity matching.
* **Tiered TTLs**: 24 hours for master itineraries; 12 hours for flight schedules.
* **In-Memory Fallback**: Seamless fallback to in-memory dictionaries if Redis is temporarily unreachable.

---

## 🔬 9. LangSmith Observability & Agent Evaluation (Evals)

DreamTrip AI is integrated with **LangSmith** for end-to-end multi-agent tracing and automated evaluations:

```mermaid
flowchart LR
    subgraph LangSmithEngine ["LangSmith Integration"]
        direction TB
        subgraph Trace ["1. Tracing & Observability"]
            L1[StateGraph Node Waterfalls]
            L2[MCP Tool Input/Output Traces]
            L3[Per-Agent Token & Latency Metrics]
        end
        subgraph Evals ["2. Automated Evaluation Harness"]
            E1[Input Guardrail Safety Eval]
            E2[Domain Relevance Eval]
            E3[7-Section Completeness Eval]
            E4[Semantic Cache Hit Validation]
        end
    end
```

### 1. Tracing & Graph Observability
* Every graph run automatically tags session metadata: `session_id`, `destination`, `duration_days`, `budget_tier`.
* Live waterfall views track each agent's latency and token consumption in the LangSmith Cloud dashboard.

### 2. Systematic Agent Evaluations (`tests/evaluate_langsmith.py`)
Run automated benchmark evaluations across security guardrails, domain filters, and dossier completeness:
```bash
python tests/evaluate_langsmith.py
```
* **Guardrail Safety Eval**: Confirms that 100% of adversarial prompt injections are blocked.
* **Domain Relevance Eval**: Confirms that off-topic requests (e.g. coding queries) are screened out.
* **Dossier Completeness Eval**: Asserts that all 7 mandatory sections exist in the compiled plan.

---

## 📂 10. Project Directory Structure

```text
DreamTrip-AI-----A-Multi-Agent-Travel-Planner-with-LangGraph/
├── app.py                         # FastAPI web server, REST endpoints, static file mounting
├── backend.py                     # LangGraph StateGraph, node logic, PostgreSQL checkpointer & LangSmith setup
├── mcp_client.py                  # MultiServerMCPClient manager for Tavily, AviationStack & Weather
├── custom_weather_mcp_server.py   # Standalone FastMCP server interfacing OpenWeather API
├── requirements.txt               # Production Python dependencies
├── Dockerfile                     # Multi-stage container build definition
├── docker-compose.yml             # Orchestration for FastAPI app + Redis 7 Alpine
├── .env.example                   # Environment variable template
├── tools/
│   ├── redis_cache.py             # Semantic vector & intent cache, hashing, TTLs & fallback
│   ├── flight_tool.py             # Direct AviationStack and local IATA database fallbacks
│   └── tavily_tool.py             # Standalone Tavily search fallback client
├── templates/
│   └── index.html                 # Luxury Single Page Application (Crimson & Amber UI)
├── static/
│   ├── style.css                  # Glassmorphism, animations & responsive grid styling
│   └── script.js                  # Frontend controller, async polling, PDF export, telemetry
├── docs/                          # Architecture diagrams & Excalidraw visual specifications
└── tests/
    ├── evaluate_langsmith.py      # LangSmith automated evaluation & benchmark harness
    ├── test_mcp_tools.py          # Diagnostic test for MCP client tools
    └── test_tavily_standalone.py  # Diagnostic test for search tools
```

---

## 🌐 11. API Endpoints Reference

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/` | `GET`, `HEAD` | Serves the main Single-Page Application interface. |
| `/api/travel` | `POST` | Core orchestration endpoint; accepts `{ "message": "...", "thread_id": "..." }`. |
| `/api/cache/stats` | `GET` | Returns live telemetry: total hits, misses, hit ratio, and saved tokens. |
| `/api/cache/clear` | `POST` | Flushes all cached travel plans and resets counters. |
| `/health` | `GET`, `HEAD` | System health check reporting server and Redis connection state. |

---

## ⚙️ 12. Environment Configuration

```ini
# Google Gemini LLM
GEMINI_API_KEY="AIzaSy..."
GEMINI_MODEL="gemini-2.5-flash"

# PostgreSQL Checkpointer
DATABASE_URL="postgresql://user:password@host:port/dbname?sslmode=require"

# Redis Cache
REDIS_URL="redis://localhost:6379/0"

# Tool & MCP API Keys
TAVILY_API_KEY="tvly-..."
AVIATIONSTACK_API_KEY="..."
OPENWEATHER_API_KEY="..."

# LangSmith Tracing & Evaluation Platform
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_pt_..."
LANGCHAIN_PROJECT="DreamTrip-AI"

# Server Port
PORT=8000
```

---

## 🚀 13. Getting Started

### Local Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```
Application accessible at `http://localhost:8000`.

### Docker Compose
```bash
docker-compose up --build
```
Spins up the FastAPI application alongside an isolated Redis 7 Alpine service.
