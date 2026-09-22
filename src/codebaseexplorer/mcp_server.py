from pathlib import Path

from fastmcp import FastMCP

from .agent import Answer, search


def start_mcp():
    mcp = FastMCP(__package__.capitalize() if __package__ != None else "CodebaseExplorer")
    mcp.add_tool(search_tool)
    mcp.run()

async def search_tool(query: str, project_dir: Path) -> Answer:
    """Ответить на вопрос по подключённому проекту с источниками и ограничениями."""
    return await search(query, project_dir)

'''
def start_mcp(
    project_dir: Path,
    *,
    ollama_url: str = "http://localhost:11434",
    chat_model: str = CHAT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
) -> None:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    mcp = FastMCP(__package__.capitalize() if __package__ is not None else "CodebaseExplorer")
    agent = create_agent(
        project_dir,
        ollama_url=ollama_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
    )

    @mcp.tool
    async def search_tool(query: str) -> AgentAnswer:
        """Ответить на вопрос по подключённому проекту с источниками и ограничениями."""
        return await agent.ask(query)

    mcp.run()

def search_tool(query: str) -> AgentAnswer:
    return run_agent(query)
'''
