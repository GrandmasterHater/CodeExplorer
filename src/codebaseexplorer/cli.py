from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import RunUsage, UsageLimits

from .agent import Dependencies, create_agent, create_answer_agent
from .infrastructure.ollama_client import CHAT_MODEL, EMBEDDING_MODEL, Ollama
from .infrastructure.semantic_provider import LocalSemanticProvider
from .initializer.indexer import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local codebase explorer")

    parser.add_argument("command", choices=["index", "search", "ask"])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--chat-model", default=CHAT_MODEL)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)

    args = parser.parse_args()

    if args.command != "index" and not args.query.strip():
        parser.error("--query is required for search and ask")

    return args


def print_run_trace(messages: list[ModelMessage]) -> None:
    print("\nAgent trace (truncated; may contain source code):", file=sys.stderr)
    request_number = 0

    for message in messages:
        if isinstance(message, ModelResponse):
            request_number += 1
            print(
                f"[Response {request_number}] finish_reason={message.finish_reason} "
                f"parts={[part.part_kind for part in message.parts]} "
                f"usage={message.usage}",
                file=sys.stderr,
            )

        for part in message.parts:
            if isinstance(part, ToolCallPart):
                detail = f"call {part.tool_name} [{part.tool_call_id}]: {part.args!r}"
            elif isinstance(part, ToolReturnPart):
                detail = f"return {part.tool_name} [{part.tool_call_id}]: {part.content!r}"
            elif isinstance(part, RetryPromptPart):
                detail = f"retry {part.tool_name or '(output)'}: {part.content!r}"
            elif isinstance(part, TextPart):
                detail = f"text: {part.content!r}"
            else:
                continue

            print(f"  {detail[:1000]}{' …' if len(detail) > 1000 else ''}", file=sys.stderr)


async def run(args: argparse.Namespace) -> None:
    root = args.project.expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"Project directory does not exist: {root}")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=10.0),
    ) as http:
        ollama = Ollama(
            http,
            base_url=args.ollama,
            chat_model=args.chat_model,
            embedding_model=args.embedding_model,
        )
        provider = LocalSemanticProvider(root, ollama)

        try:
            if args.command == "index":
                await build_index(
                    root,
                    provider,
                    ollama,
                    rebuild=args.rebuild,
                    no_cache=args.no_cache,
                )
                print("Index is ready.")
                return

            if args.command == "search":
                hits = await provider.search(args.query, limit=5)

                for hit in hits:
                    print(
                        f"\n[{hit.kind}] {hit.path}:"
                        f"{hit.start_line}-{hit.end_line}"
                    )
                    print(f"Symbol: {hit.name or '(file)'}")
                    print(f"Score: {hit.score}")
                    print(hit.description)

                if not hits:
                    print("No results.")

                return

            model = OllamaModel(
                args.chat_model,
                provider=OllamaProvider(
                    base_url=f"{args.ollama.rstrip('/')}/v1",
                ),
            )

            agent = create_agent(model)
            usage = RunUsage()

            with capture_run_messages() as messages:
                try:
                    evidence = await agent.run(
                        args.query,
                        deps=Dependencies(
                            project_dir=root,
                            semantic_provider=provider,
                        ),
                        # Reserve two of the eight requests for the structured answer.
                        usage=usage,
                        usage_limits=UsageLimits(
                            request_limit=6,
                            tool_calls_limit=12,
                        ),
                        model_settings={
                            "temperature": 0.1,
                            "max_tokens": 2000,
                            "timeout": 180.0,
                        },
                    )
                except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
                    print_run_trace(messages)
                    if exc.__cause__ is not None:
                        print(f"Cause: {str(exc.__cause__)[:2000]}", file=sys.stderr)
                    raise

            answer_agent = create_answer_agent(model)
            with capture_run_messages() as messages:
                try:
                    result = await answer_agent.run(
                        args.query,
                        message_history=evidence.all_messages(),
                        usage=usage,
                        usage_limits=UsageLimits(request_limit=8, tool_calls_limit=12),
                        model_settings={
                            "temperature": 0,
                            "max_tokens": 2000,
                            "timeout": 180.0,
                        },
                    )
                except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
                    print_run_trace(messages)
                    if exc.__cause__ is not None:
                        print(f"Cause: {str(exc.__cause__)[:2000]}", file=sys.stderr)
                    raise

            answer = result.output
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

            print(f"\nUsage: {result.usage}")

        finally:
            await provider.close()


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
