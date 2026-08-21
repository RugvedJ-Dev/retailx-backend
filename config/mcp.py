from dotenv import load_dotenv
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
import asyncio
import threading

load_dotenv()

MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN")
MCP_SERVER_URI = os.getenv("MCP_SERVER_URI")

client = MultiServerMCPClient({
    # For HTTP transport
    "fastmcp_http_server": {
        "transport": "http",
        "url": f"{MCP_SERVER_URI}",
        "headers": {
            "Authorization": f"Bearer {MCP_BEARER_TOKEN}",
            "Content-Type": "application/json",
        },
    },
}
)


_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    return _submit_async(coro)



def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []