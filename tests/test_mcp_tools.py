"""
DreamTrip AI — MCP Tool Discovery Test
"""
import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
from mcp_client import get_all_tools

if __name__ == "__main__":
    asyncio.run(get_all_tools())
