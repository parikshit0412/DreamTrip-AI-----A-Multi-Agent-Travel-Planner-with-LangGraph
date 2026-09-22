"""
================================================================================
DreamTrip AI — Multi-Agent Travel Planner Engine (LangGraph Backend)
================================================================================
This module defines the autonomous multi-agent orchestration architecture
powered by LangGraph StateGraph, Google Gemini 2.5 Flash, Model Context Protocol
(MCP) tools, PostgreSQL checkpointing, and Redis token conservation caching.

Multi-Agent Pipeline Execution Flow:
    [START]
       │
       ▼
    flight_agent    ──> Queries AviationStack MCP tool for flights/airports/routes
       │
       ▼
    hotel_agent     ──> Queries Tavily AI Search MCP tool for stays/resorts
       │
       ▼
    weather_agent   ──> Extracts destination & queries OpenWeather MCP tool for live/forecast weather
       │
       ▼
    itinerary_agent ──> Gemini 2.5 Flash: synthesizes chronological day-by-day itinerary
       │
       ▼
    final_agent     ──> Gemini 2.5 Flash: compiles structured executive travel master plan
       │
       ▼
     [END]
================================================================================
"""

# ==============================================================================
# 1. System, Security, & Environment Initialization
# ==============================================================================
import os          # Operating system interface: reads environment variables & filesystem paths
import certifi     # Authoritative collection of Mozilla root CA certificates for trusted TLS/SSL
import warnings    # Runtime warning filtering subsystem
import asyncio     # Asynchronous I/O event loop support for executing async MCP client tools
import operator    # Standard operators (operator.add) used as state reducers in LangGraph
import uuid        # Universally unique identifier generator for session/thread tracking
from typing import TypedDict, Annotated  # Type annotations: TypedDict for state schema, Annotated for reducers
from dotenv import load_dotenv           # Loads environment variables from the local .env file

# Load environment configuration from .env into os.environ
load_dotenv()

# Filter harmless Google Generative AI SDK deprecation and function-calling warnings
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*google.*")

# Enforce certified Mozilla CA certificate bundles to prevent Windows SSL handshake errors
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ==============================================================================
# 2. Database & LangChain / LangGraph Orchestration Libraries
# ==============================================================================
import psycopg                           # PostgreSQL client driver for state persistence
from psycopg.rows import dict_row        # Row factory returning database query rows as standard Python dicts
from langchain_google_genai import ChatGoogleGenerativeAI  # LangChain integration for Google Gemini LLM
from langgraph.graph import StateGraph, START, END         # Core LangGraph graph constructor & boundary nodes
from langgraph.checkpoint.postgres import PostgresSaver    # PostgreSQL persistent state checkpointer
from langchain_core.messages import (                      # Canonical message representations across nodes
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

# High-performance Redis caching engine to conserve LLM tokens and reduce API latency
from tools.redis_cache import cache

# Asynchronous Model Context Protocol (MCP) tool bindings and destination extractor
from mcp_client import (
    tavily_mcp_search,      # MCP client function to query Tavily search for hotels & locations
    aviation_mcp_call,      # MCP client function to query AviationStack for flights & airports
    weather_mcp_search,     # MCP client function to fetch current weather via OpenWeather MCP
    forecast_mcp_search,    # MCP client function to fetch 5-day weather forecast via OpenWeather MCP
    extract_destination,    # LLM-assisted destination name extraction helper
)


# ==============================================================================
# 3. Database URL & Configuration Resolver
# ==============================================================================
def get_database_url() -> str:
    """
    Retrieves and validates the PostgreSQL connection URL from environment variables.
    Ensures 'sslmode=require' is appended for secure, encrypted cloud database transit.
    """
    # Read DATABASE_URL string from environment variables
    database_url = os.getenv("DATABASE_URL")

    # Raise explicit configuration error if the database connection string is absent
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your PostgreSQL database URL to .env"
        )

    # Automatically enforce SSL encryption for cloud PostgreSQL instances (e.g. Render, Supabase, Neon)
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# ==============================================================================
# 4. Google Gemini LLM Client Initialization
# ==============================================================================
# Retrieve Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate presence of the Gemini API key
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please add your Gemini API Key to .env"
    )

# Configurable Gemini model identifier (defaults to 'gemini-2.5-flash' for reliable daily quota)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Instantiate the primary Gemini chat model instance
# Temperature is set to 0 to ensure deterministic, accurate, and reproducible travel itineraries
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0
)


def extract_text(content) -> str:
    """
    Safely normalizes response content into a clean string, handling both raw string
    responses and multi-part dictionary structures returned by Google Gemini SDK.
    """
    # Direct return if content is already a plain string
    if isinstance(content, str):
        return content
    # Concatenate text segments if content is provided as a list of parts/dictionaries
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    # Fallback conversion to string
    return str(content)


# ==============================================================================
# 5. LangGraph Centralized State Schema
# ==============================================================================
class TravelState(TypedDict):
    """
    Defines the centralized state schema passed and modified across all agent nodes.
    - Fields with `Annotated[..., operator.add]` accumulate additions across node returns.
    - Non-annotated fields overwrite previous values with the newest node output.
    """
    messages: Annotated[list[AnyMessage], operator.add]  # Accumulating message history across nodes
    user_query: str                                      # Original travel request provided by user
    flight_results: str                                  # Flight & airport intelligence gathered by Flight Agent
    hotel_results: str                                   # Curated hotel & accommodation options from Hotel Agent
    weather_results: str                                 # Real-time weather and forecast data from Weather Agent
    itinerary: str                                       # Draft day-by-day travel schedule from Itinerary Agent
    llm_calls: Annotated[int, operator.add]              # Monotonically accumulated counter of LLM invocations


# ==============================================================================
# 6. Agent Node 1: Flight Intelligence Agent
# ==============================================================================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

def flight_agent(state: TravelState):
    """
    LangGraph Node: Discovers flight routes, airports, and airline schedules.
    Leverages AviationStack MCP tool endpoints and synthesizes guidance using Gemini.
    """
    # Extract original user query from current graph state
    query = state["user_query"]

    try:
        # Asynchronously fetch airport list from AviationStack MCP tool
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        # Asynchronously fetch airline list from AviationStack MCP tool
        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        # Format prompt with retrieved airport & airline context (bounded to 3000 chars each)
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        # Invoke Gemini model with flight planning persona
        response = llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt)
        ])

        # Normalize extracted text output
        flight_data = extract_text(response.content)

    except Exception as e:
        # Graceful fallback in case of API or network failure
        flight_data = f"Flight information unavailable: {str(e)}"

    # Return partial state update
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight recommendations generated.")
        ],
        "llm_calls": 1  # Record 1 LLM call
    }


# ==============================================================================
# 7. Agent Node 2: Hotel & Stays Discovery Agent
# ==============================================================================
def hotel_agent(state: TravelState):
    """
    LangGraph Node: Discovers curated hotels, resorts, and accommodations.
    Leverages Tavily AI Search MCP tool for real-time web intelligence.
    """
    # Formulate search query targeting top accommodations for the user query
    query = f"Best hotels for {state['user_query']}"

    try:
        # Asynchronously query Tavily search MCP tool
        hotel_results = asyncio.run(tavily_mcp_search(query))
    except Exception as e:
        # Graceful fallback if Tavily search fails
        hotel_results = f"Hotel information unavailable: {str(e)}"

    # Return partial state update
    return {
        "hotel_results": str(hotel_results),
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": 0  # 0 LLM calls (pure tool search)
    }


# ==============================================================================
# 8. Agent Node 3: Weather Intelligence Agent
# ==============================================================================
def weather_agent(state: TravelState):
    """
    LangGraph Node: Extracts the destination city and retrieves real-time weather
    and 5-day weather forecasts using the OpenWeather MCP tool server.
    """
    try:
        # Extract destination city/country from user query using LLM extractor
        city = extract_destination(state["user_query"])

        # Query real-time current weather data via OpenWeather MCP tool
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        # Query multi-day forecast data via OpenWeather MCP tool
        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        # Compile formatted weather results string
        weather_results = f"Current Weather:\n{weather_data}\n\nForecast:\n{forecast_data}"

    except Exception as e:
        # Graceful fallback in case of MCP weather server or API error
        weather_results = f"Weather information unavailable: {str(e)}"

    # Return partial state update
    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(content="Weather information fetched.")
        ],
        "llm_calls": 1  # 1 LLM call spent on destination extraction
    }


# ==============================================================================
# 9. Agent Node 4: Sequential Itinerary Planner Agent
# ==============================================================================
def itinerary_agent(state: TravelState):
    """
    LangGraph Node: Synthesizes a practical, chronological day-by-day travel itinerary.
    Consumes flight schedules, curated stays, and local weather forecasts to draft
    logistically sound activities and timing.
    """
    # Build comprehensive prompt combining flight, hotel, and weather intelligence
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

Make the itinerary practical, budget-aware, weather-appropriate, and easy to follow.
"""

    # Invoke Gemini model to draft detailed day-by-day itinerary
    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    # Return partial state update
    return {
        "itinerary": extract_text(response.content),
        "messages": [response],
        "llm_calls": 1  # Record 1 LLM invocation
    }


# ==============================================================================
# 10. Agent Node 5: Final Executive Synthesis Agent
# ==============================================================================
def final_agent(state: TravelState):
    """
    LangGraph Node: Synthesizes all gathered intelligence into an executive master plan.
    Structures sections: Summary, Flights, Hotels, Weather, Itinerary, Budget, and Tips.
    """
    # Build master compilation prompt
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear, practical, and highly organized.
- Mention that live flight API may not provide exact ticket prices if pricing is unavailable.
- Include weather-based travel advice and packing tips.
- Keep the response useful, realistic, and actionable for real travel planning.
"""

    # Invoke Gemini model to synthesize master executive plan
    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    # Return partial state update
    return {
        "messages": [response],
        "llm_calls": 1  # Record 1 LLM invocation
    }


# ==============================================================================
# 11. LangGraph Multi-Agent Architecture Assembly
# ==============================================================================
# Instantiate the StateGraph initialized with the typed TravelState schema
graph = StateGraph(TravelState)

# 1. Register all 5 specialized agent nodes
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# 2. Sequential pipeline edge routing:
# START -> flight_agent -> hotel_agent -> weather_agent -> itinerary_agent -> final_agent -> END
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# ==============================================================================
# 12. PostgreSQL Persistent State Checkpointer Setup
# ==============================================================================
# Resolve secure PostgreSQL connection URL
DATABASE_URL = get_database_url()

# Establish direct database connection with dictionary-formatted row output
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

# PostgresSaver persists graph states, enabling session recovery, conversation history, and time-travel
checkpointer = PostgresSaver(_conn)
checkpointer.setup()  # Automatically creates required checkpoint database tables if not existing

# Compile graph into an executable runnable backed by PostgreSQL state memory checkpointer
travel_graph = graph.compile(checkpointer=checkpointer)


# ==============================================================================
# 13. High-Level Orchestration Entry Point
# ==============================================================================
def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict:
    """
    Main entry point invoked by the FastAPI web endpoint:
    1. Checks Redis cache for an identical or normalized query.
       - Cache HIT: Returns instantly (< 10ms), conserving 100% of LLM calls and tokens.
    2. Cache MISS: Executes the 5-node LangGraph StateGraph pipeline backed by PostgreSQL state.
    3. Caches the synthesized master plan in Redis with a 24-hour TTL (86,400 seconds).
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
            "weather_results": cached_plan.get("weather_results", ""),
            "itinerary": cached_plan.get("itinerary", ""),
            "llm_calls": 0,  # Zero LLM calls required when served from cache
            "is_cached": True,
            "cache_source": backend_name,
            "tokens_saved": tokens_saved,
        }

    # --------------------------------------------------------------------------
    # Step 2: Cache Miss — Execute Multi-Agent LangGraph Workflow
    # --------------------------------------------------------------------------
    # Generate unique session thread ID if one is not provided
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    # Configure session thread ID for PostgreSQL state checkpointer
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Invoke the compiled multi-agent state graph with initial state payload
    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    # Extract final synthesized response and agent outputs from completed state
    final_answer = extract_text(result["messages"][-1].content)
    flight_results = result.get("flight_results", "")
    hotel_results = result.get("hotel_results", "")
    weather_results = result.get("weather_results", "")
    itinerary = result.get("itinerary", "")
    llm_calls = result.get("llm_calls", 3)

    # Estimate total token consumption across prompts, responses, and agent context
    total_text_chars = (
        len(user_input)
        + len(flight_results)
        + len(hotel_results)
        + len(weather_results)
        + len(itinerary)
        + len(final_answer)
    )
    estimated_tokens = max(3800, round(total_text_chars / 3.6))

    # --------------------------------------------------------------------------
    # Step 3: Store in Redis Cache with 24-Hour TTL (86,400 seconds)
    # --------------------------------------------------------------------------
    plan_data = {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_results,
        "hotel_results": hotel_results,
        "weather_results": weather_results,
        "itinerary": itinerary,
        "estimated_tokens": estimated_tokens,
    }
    cache.set_cached_plan(user_input, plan_data, ttl_seconds=86400)

    # Return complete orchestrated response dictionary
    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_results,
        "hotel_results": hotel_results,
        "weather_results": weather_results,
        "itinerary": itinerary,
        "llm_calls": llm_calls,
        "is_cached": False,
        "cache_source": None,
        "tokens_saved": 0,
    }