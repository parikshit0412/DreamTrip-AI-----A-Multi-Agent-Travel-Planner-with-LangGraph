"""
================================================================================
DreamTrip AI — Multi-Agent Travel Planner Engine (LangGraph Backend)
================================================================================
This module defines the autonomous multi-agent orchestration architecture
powered by LangGraph StateGraph, Google Gemini 2.5 Flash, PostgreSQL checkpointing,
and Redis token conservation caching.

Pipeline Flow:
    [START] 
       ├──> flight_agent (Queries AviationStack API + Redis cache)
       └──> hotel_agent  (Queries Tavily Search API + Redis cache)
               │
               ▼
         itinerary_agent (Gemini 2.5 Flash: drafts day-by-day itinerary)
               │
               ▼
           final_agent   (Gemini 2.5 Flash: synthesizes executive master plan)
               │
             [END]
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. System & Security Libraries
# ------------------------------------------------------------------------------
import os          # Operating system interface: reads environment variables (.env) & paths
import certifi     # Authoritative collection of Mozilla root CA certificates for trusted SSL/TLS
import warnings    # Control warnings filtering (suppresses non-critical Google SDK warnings)
from dotenv import load_dotenv  # Automatically loads environment variables from local .env file

# Load environment configuration (.env) into process memory
load_dotenv()

# Filter harmless deprecation / automatic function calling warnings from Google Generative AI
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*google.*")

# Force Python HTTP clients to use certified CA bundles (prevents SSL Handshake errors on Windows)
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ------------------------------------------------------------------------------
# 2. Type Hints & Functional Utilities
# ------------------------------------------------------------------------------
from typing import TypedDict, Annotated  # TypedDict defines state contracts; Annotated attaches reducers
import operator                          # Supplies operator.add to accumulate messages & counters in state
import uuid                              # Generates unique thread/session IDs for persistent conversational memory

# ------------------------------------------------------------------------------
# 3. Database & LangChain / LangGraph Orchestration Libraries
# ------------------------------------------------------------------------------
import psycopg                           # PostgreSQL client for Python (v3), used for state checkpointer
from psycopg.rows import dict_row        # Row factory returning database query rows as Python dictionaries
from langchain_google_genai import ChatGoogleGenerativeAI  # LangChain model wrapper for Google Gemini 2.5 Flash

from langgraph.graph import StateGraph, START, END         # Core LangGraph graph builder and boundary nodes
from langgraph.checkpoint.postgres import PostgresSaver    # Persistent checkpointing engine storing state in Postgres
from langchain_core.messages import (    # Canonical message representations across LangChain nodes
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

# ------------------------------------------------------------------------------
# 4. Custom Tools & Redis Cache Subsystem
# ------------------------------------------------------------------------------
from tools.tavily_tool import tavily_search  # Hotel and destination discovery using Tavily AI Search
from tools.flight_tool import search_flights # Flight route schedule discovery using AviationStack API
from tools.redis_cache import cache          # High-performance Redis caching engine to conserve LLM tokens


def get_database_url() -> str:
    """
    Retrieves and validates the PostgreSQL connection URL from environment variables.
    Appends 'sslmode=require' if missing to ensure secure encrypted database transit.
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your PostgreSQL database URL to .env"
        )

    # Ensure SSL encryption is enabled for cloud PostgreSQL instances (e.g. Render, Supabase, Neon)
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# Retrieve Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add your Gemini API Key to .env"
    )

# Instantiate the primary Gemini 2.5 Flash model with deterministic temperature (0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0  # Zero temperature ensures consistent, reproducible travel planning outputs
)


# ==============================================================================
# Graph State Definition
# ==============================================================================
class TravelState(TypedDict):
    """
    Defines the centralized state schema passed across all nodes in the LangGraph StateGraph.
    Uses 'Annotated[..., operator.add]' for fields that should be accumulated rather than overwritten.
    """
    messages: Annotated[list[AnyMessage], operator.add]  # Accumulating message history
    user_query: str                                      # The original travel query from the user
    flight_results: str                                  # Intelligence retrieved by the Flight Agent
    hotel_results: str                                   # Stays and hotel options retrieved by the Hotel Agent
    itinerary: str                                       # Draft day-by-day schedule drafted by Itinerary Agent
    llm_calls: Annotated[int, operator.add]              # Monotonically increasing counter of LLM invocations


# Exported tools list
tools = [tavily_search, search_flights]


# ==============================================================================
# Agent Node 1: Flight Intelligence Agent
# ==============================================================================
def flight_agent(state: TravelState):
    """
    LangGraph Node: Discovers flight routes, airports, and schedules.
    Leverages AviationStack API with transparent Redis query caching.
    """
    query = state["user_query"]
    flight_data = search_flights(query)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": 1
    }


# ==============================================================================
# Agent Node 2: Hotel & Stays Discovery Agent
# ==============================================================================
def hotel_agent(state: TravelState):
    """
    LangGraph Node: Discovers curated hotels, resorts, and guesthouses.
    Leverages Tavily AI Search with transparent Redis query caching.
    """
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": 1
    }




# ==============================================================================
# Agent Node 3: Sequential Itinerary Planner Agent
# ==============================================================================
def itinerary_agent(state: TravelState):
    """
    LangGraph Node: Synthesizes a practical, chronological day-by-day travel itinerary.
    Consumes both flight arrival/departure schedules and curated hotel locations to
    generate realistic logistical timing.
    """
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    # Invocates Gemini 2.5 Flash with system role instructions and compiled context
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": 1
    }


# ==============================================================================
# Agent Node 4: Final Executive Synthesis Agent
# ==============================================================================
def final_agent(state: TravelState):
    """
    LangGraph Node: Formats the entire travel brief into an executive master plan.
    Structures sections: Summary, Flights, Hotels, Itinerary, Budget Breakdown, and Tips.
    """
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": 1
    }


# ==============================================================================
# LangGraph Multi-Agent Architecture Assembly
# ==============================================================================
# Initialize the StateGraph with the typed TravelState schema
graph = StateGraph(TravelState)

# 1. Register all agent nodes
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# 2. Parallel fan-out from START to Flight Agent and Hotel Agent
graph.add_edge(START, "flight_agent")
graph.add_edge(START, "hotel_agent")

# 3. Fan-in: Both Flight and Hotel intelligence must be ready before Itinerary starts
graph.add_edge("flight_agent", "itinerary_agent")
graph.add_edge("hotel_agent", "itinerary_agent")

# 4. Sequential execution to Final Synthesis and END
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# ==============================================================================
# PostgreSQL Persistent Checkpointer Setup
# ==============================================================================
DATABASE_URL = get_database_url()

# Establish connection to PostgreSQL database with dictionary row formatting
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

# PostgresSaver persists graph states, enabling session recovery, time-travel, and history
checkpointer = PostgresSaver(_conn)
checkpointer.setup()  # Creates checkpoints tables if they don't already exist

# Compile graph into an executable runnable backed by PostgreSQL state memory
travel_graph = graph.compile(checkpointer=checkpointer)


# ==============================================================================
# FastAPI Orchestration Service Function
# ==============================================================================
def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict:
    """
    Main entry point invoked by the FastAPI endpoint:
    1. Checks Redis cache for an identical or normalized query.
       - Cache HIT: Returns instantly (< 10ms) saving 100% of LLM calls and tokens.
    2. Cache MISS: Invokes the LangGraph StateGraph, compiling agents and PostgreSQL state.
    3. Caches the synthesized master plan in Redis with a 24-hour TTL.
    """
    # --------------------------------------------------------------------------
    # Step 1: Check Redis Token Conservation Cache
    # --------------------------------------------------------------------------
    cached_plan = cache.get_cached_plan(user_input)
    if cached_plan:
        tokens_saved = cached_plan.get("estimated_tokens", 4500)
        cache.record_token_savings(tokens_saved)
        backend_name = "Redis" if cache.is_connected else "In-Memory Fallback"
        print(f"[{backend_name} Cache HIT] Serving plan for '{user_input[:40]}...' (Saved ~{tokens_saved} tokens)")
        return {
            "thread_id": thread_id or cached_plan.get("thread_id", f"user_{uuid.uuid4().hex}"),
            "answer": cached_plan.get("answer", ""),
            "flight_results": cached_plan.get("flight_results", ""),
            "hotel_results": cached_plan.get("hotel_results", ""),
            "itinerary": cached_plan.get("itinerary", ""),
            "llm_calls": 0,  # Zero LLM calls required when served from cache
            "is_cached": True,
            "cache_source": backend_name,
            "tokens_saved": tokens_saved,
        }

    # --------------------------------------------------------------------------
    # Step 2: Cache Miss — Execute Multi-Agent LangGraph Workflow
    # --------------------------------------------------------------------------
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id  # Session ID tracked by PostgreSQL checkpointer
        }
    }

    # Invoke the compiled multi-agent state graph
    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content
    flight_results = result.get("flight_results", "")
    hotel_results = result.get("hotel_results", "")
    itinerary = result.get("itinerary", "")
    llm_calls = result.get("llm_calls", 2)

    # Estimate total tokens generated across prompts, itineraries, and final response
    total_text_chars = len(user_input) + len(flight_results) + len(hotel_results) + len(itinerary) + len(final_answer)
    estimated_tokens = max(3800, round(total_text_chars / 3.6))

    # --------------------------------------------------------------------------
    # Step 3: Store in Redis Cache with 24-Hour TTL (86,400 seconds)
    # --------------------------------------------------------------------------
    plan_data = {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_results,
        "hotel_results": hotel_results,
        "itinerary": itinerary,
        "estimated_tokens": estimated_tokens,
    }
    cache.set_cached_plan(user_input, plan_data, ttl_seconds=86400)

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_results,
        "hotel_results": hotel_results,
        "itinerary": itinerary,
        "llm_calls": llm_calls,
        "is_cached": False,
        "cache_source": None,
        "tokens_saved": 0,
    }