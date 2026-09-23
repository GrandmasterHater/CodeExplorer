from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

from .agent import search
from .infrastructure.ollama_client import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    Ollama,
)
from .infrastructure.semantic_provider import LocalSemanticProvider
from .initializer.indexer import build_index
from .mcp_server import start_mcp
from .utils.print_utils import print_run_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local codebase explorer")

    parser.add_argument("command", choices=["index", "search", "mcp"])
    parser.add_argument("--project", type=Path)
    parser.add_argument("--query", default="")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    if args.command in {"search"} and not args.query.strip():
        parser.error("--query is required for search")

    if args.command in {"search", "index"} and args.project is None:
        parser.error("–project is required for index/search")

    return args


async def _index_cmd(root: Path, is_rebuild: bool, is_no_cache: bool):
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0),) as http:
        ollama = Ollama(
            http,
            base_url=OLLAMA_URL,
            chat_model=CHAT_MODEL,
            embedding_model=EMBEDDING_MODEL,
        )
        provider = LocalSemanticProvider(root, ollama)

        try:
            await build_index(
                root,
                provider,
                ollama,
                rebuild=is_rebuild,
                no_cache=is_no_cache,
            )
            print("Index is ready.")
            return
        finally:
            await provider.close()

async def _search_cmd(query: str, root: Path):
    answer = await search(query, root)

    print(answer.text)

    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            location = source.path
            if source.start_line is not None:
                location += f":{source.start_line}"
                if source.end_line is not None:
                    location += f"-{source.end_line}"
            print(f"- {location}")

    if answer.limitations:
        print("\nLimitations:")
        for limitation in answer.limitations:
            print(f"- {limitation}")


async def run(args: argparse.Namespace) -> None:
    if args.command == "mcp":
        await start_mcp()
        return

    root = args.project.expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"Project directory does not exist: {root}")

    if args.command == "index":
        await _index_cmd(root, args.rebuild, args.no_cache)
        return

    if args.command == "search":
        await _search_cmd(args.query, root)
        return


def main() -> None:
    args = parse_args()

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
