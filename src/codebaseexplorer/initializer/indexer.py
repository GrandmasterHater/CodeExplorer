from __future__ import annotations

import hashlib
import json
from pathlib import Path

from codebaseexplorer.infrastructure.ollama_client import Ollama
from codebaseexplorer.infrastructure.semantic_provider import (
    COLLECTION,
    LocalSemanticProvider,
    SearchHit,
)

from .parser import find_code_files, parse_file

PIPELINE_VERSION = "v1"
CHUNK_CHARS = 4000


def split_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if size < 1:
        raise ValueError("Chunk size must be positive")

    parts = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline >= start + size // 2:
                end = newline + 1

        parts.append(text[start:end])
        start = end

    return parts


async def summarize(
    ollama: Ollama,
    text: str,
    *,
    context: str,
) -> str:
    if not text.strip():
        return "The source is empty or contains only whitespace."

    chunks = split_text(text)
    summaries = []

    for number, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"  Part {number}/{len(chunks)}", flush=True)

        summaries.append(
            await ollama.describe(
                f"Context:\n{context}\n\n"
                f"Source fragment {number}/{len(chunks)}:\n"
                f"{chunk}"
            )
        )

    # Сводим описания небольшими группами, а не отправляем их все
    # одним потенциально огромным запросом.
    while len(summaries) > 1:
        combined = []

        for offset in range(0, len(summaries), 4):
            group = summaries[offset : offset + 4]

            if len(group) == 1:
                combined.append(group[0])
                continue

            combined.append(
                await ollama.describe(
                    f"Context:\n{context}\n\n"
                    "Merge the following partial descriptions. "
                    "Preserve distinct responsibilities and important names. "
                    "These descriptions may be incomplete or imperfect.\n\n"
                    + "\n\n".join(group)
                )
            )

        summaries = combined

    return summaries[0]


async def build_index(
    root: Path,
    provider: LocalSemanticProvider,
    ollama: Ollama,
    *,
    rebuild: bool,
    no_cache: bool,
) -> None:
    if await provider.qdrant_client.collection_exists(COLLECTION) and not rebuild:
        raise RuntimeError("Index already exists. Add --rebuild to replace it.")

    paths = find_code_files(root)
    if not paths:
        raise RuntimeError("No .cs or .java files found in the selected directory")

    print(f"Found {len(paths)} source files.", flush=True)

    cache_dir = root / ".explorer" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[SearchHit] = []

    for number, path in enumerate(paths, start=1):
        relative_path = path.relative_to(root).as_posix()
        print(f"[{number}/{len(paths)}] {relative_path}", flush=True)

        text, source_hash, symbols = parse_file(path)

        cache_identity = (
            f"{PIPELINE_VERSION}\0{CHUNK_CHARS}\0"
            f"{ollama.chat_model}\0{relative_path}\0{source_hash}"
        )
        cache_key = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"

        if cache_path.exists() and not no_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            records = [SearchHit.model_validate(item) for item in cached]
            print(f"  Cache: {len(records)} records", flush=True)
        else:
            file_description = await summarize(
                ollama,
                text,
                context=f"File: {relative_path}. Describe its overall role.",
            )

            records = [
                SearchHit(
                    kind="file",
                    path=relative_path,
                    description=file_description,
                    source_hash=source_hash,
                    start_line=1,
                    end_line=max(1, len(text.splitlines())),
                )
            ]

            for index, symbol in enumerate(symbols, start=1):
                print(
                    f"  Symbol {index}/{len(symbols)}: {symbol.name}",
                    flush=True,
                )

                description = await summarize(
                    ollama,
                    symbol.code,
                    context=(
                        f"File: {relative_path}\n"
                        f"File overview: {file_description}\n"
                        f"Symbol: {symbol.name}\n"
                        "Describe this symbol, not the entire file. "
                        "The overview is context, not proof of its behavior."
                    ),
                )

                records.append(
                    SearchHit(
                        kind="function",
                        path=relative_path,
                        name=symbol.name,
                        description=description,
                        file_context=file_description,
                        source_hash=source_hash,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                    )
                )

            cache_path.write_text(
                json.dumps(
                    [record.model_dump(mode="json") for record in records],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Не смешиваем описание старого содержимого с новым файлом,
        # если исходники изменили во время долгой индексации.
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError(f"File changed during indexing: {relative_path}")

        all_records.extend(records)

    records_path = root / ".explorer" / "records.jsonl"
    records_path.write_text(
        "\n".join(record.model_dump_json() for record in all_records) + "\n",
        encoding="utf-8",
    )

    print(f"Descriptions ready: {len(all_records)} records.", flush=True)
    print(f"Inspect descriptions at: {records_path}", flush=True)

    # Генерация описаний и embeddings разделены на две последовательные
    # фазы, чтобы не переключать модели после каждого метода.
    await provider.replace(all_records, rebuild=rebuild)
