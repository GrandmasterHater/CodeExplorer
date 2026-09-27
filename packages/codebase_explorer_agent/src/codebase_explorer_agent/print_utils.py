import sys

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


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
