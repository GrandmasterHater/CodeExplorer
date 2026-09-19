from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable  # noqa: UP035

_CODEGRAPH_NAME = "codegraph"
_CODEGRAPH_CMD_NAME = "codegraph.cmd"
_STATUS_CMD = "status"
_EXPLORE_CMD = "explore"
_SUCCESS_CODE = 0

@dataclass(frozen=True)
class FoundResult:
    query: str
    output: str


@dataclass(frozen=True)
class ErrorResult:
    query: str
    error: str


def __execute_query(func: Callable[[], subprocess.CompletedProcess[str]], operation: str) -> subprocess.CompletedProcess[str] | str:
    try:
        return func()
    except subprocess.TimeoutExpired:
        return f"CodeGraph {operation} timed out"
    except OSError as exc:
        return f"Could not run CodeGraph {operation}: {exc}"
    except UnicodeDecodeError as exc:
        return f"Could not decode CodeGraph {operation} output: {exc}"


def __build_error_info(process: subprocess.CompletedProcess[str], query: str, op: str) -> ErrorResult:
    details = (process.stderr.strip() or process.stdout.strip() or "Unknown error")
    return ErrorResult(query, f"CodeGraph {op} failed: {details}")


def explore_in_graph(project_dir: str | Path, query: str, timeout_sec: float) -> FoundResult | ErrorResult:
    if not query.strip():
        return ErrorResult(query, "Query is empty")

    resolved_dir = Path(project_dir).resolve()

    if not resolved_dir.exists():
        return ErrorResult(query, error="Path not exists")

    if not resolved_dir.is_dir():
        return ErrorResult(query, error="Path expect directory, not a file")

    #Получаем полный путь из пользовательской переменной среды текущего процесса.
    codegraph = shutil.which(_CODEGRAPH_NAME) or shutil.which(_CODEGRAPH_CMD_NAME)

    if codegraph is None:
        return ErrorResult(query, error="Codegraph CLI not found")

    req_func = lambda: subprocess.run([codegraph, _STATUS_CMD, str(resolved_dir)],
                                                        cwd=str(resolved_dir),
                                                        text=True,
                                                        capture_output=True,
                                                        timeout=timeout_sec,
                                                        check=False)
    result = __execute_query(req_func, _STATUS_CMD)

    if isinstance(result, str):
        return ErrorResult(query, error=result)

    if result.returncode != _SUCCESS_CODE:
        return __build_error_info(result, query, _STATUS_CMD)

    req_func = lambda: subprocess.run([codegraph, _EXPLORE_CMD, query],
                                      cwd=str(resolved_dir),
                                      text=True,
                                      encoding="utf-8",
                                      capture_output=True,
                                      timeout=timeout_sec,
                                      check=False)

    result = __execute_query(req_func, _EXPLORE_CMD)

    if isinstance(result, str):
        return ErrorResult(query, error=result)

    if result.returncode != _SUCCESS_CODE:
        return __build_error_info(result, query, _EXPLORE_CMD)

    return FoundResult(query, result.stdout)
