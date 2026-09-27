from pathlib import Path

from fastmcp import FastMCP

from .agent import Answer, search


async def start_mcp():
    mcp = FastMCP(__package__.capitalize() if __package__ != None else "CodebaseExplorer")
    mcp.add_tool(search_tool)
    await mcp.run_async()

async def search_tool(query: str, project_dir: Path) -> Answer:
    """Ответить на вопрос по подключённому проекту с источниками и ограничениями."""
    return await search(query, project_dir)
