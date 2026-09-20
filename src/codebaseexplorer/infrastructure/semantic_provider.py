from __future__ import annotations

import json
from functools import reduce
from pathlib import Path

# Literal ограничивает тип конкретными допустимыми значениями.
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from .ollama_client import LlmResponce, Ollama

CHAT_MODEL = "qwen3.5-9b-32k:latest"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
COLLECTION = "code_semantics_v1"


# Один результат семантического поиска.
class SearchHit(BaseModel):
    kind: Literal["file", "function"]
    path: str
    description: str
    source_hash: str
    name: str | None = None
    file_context: str = ""
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    score: float | None = None


class SemanticProvider(Protocol):
    async def exists(self) -> bool: ...

    async def search(self,query: str, *,limit: int,) -> list[SearchHit]: ...


def embedding_text(record: SearchHit) -> str:
    return (
        f"File: {record.path}\n"
        f"Symbol: {record.name or '(file)'}\n"
        f"File context: {record.file_context}\n"
        f"Behavior: {record.description}"
    )


class LocalSemanticProvider:
    def __init__(self, project_dir: Path, ollama_client: Ollama) -> None:
        self.project_dir = project_dir.resolve()
        self.ollama_client = ollama_client

        self.storage_dir = self.project_dir / ".explorer"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = self.storage_dir / "manifest.json"
        self.qdrant_client = AsyncQdrantClient(path=str(self.storage_dir / "qdrant"),)

    async def close(self) -> None:
        await self.qdrant_client.close()

    async def exists(self) -> bool:
        if not self.manifest.exists():
            return False

        metadata = json.loads(self.manifest.read_text(encoding="utf-8"))

        if not metadata.get("ready"):
            return False

        if metadata["embedding_model"] != self.ollama_client.embedding_model:
            raise RuntimeError("The index uses another embedding model. Rebuild the index.")

        return await self.qdrant_client.collection_exists(COLLECTION)

    def __write_manifest(self, *, ready: bool, records: int) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "ready": ready,
                    "records": records,
                    "chat_model": self.ollama_client.chat_model,
                    "embedding_model": self.ollama_client.embedding_model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    async def replace(
        self,
        records: list[SearchHit],
        *,
        rebuild: bool,
    ) -> None:
        if not records:
            raise ValueError("There are no records to index")

        collection_exists = await self.qdrant_client.collection_exists(COLLECTION)

        if collection_exists and not rebuild:
            raise RuntimeError(
                "Index already exists. Use --rebuild to replace it."
            )

        probe = await self.ollama_client.embed(["Embedding dimension probe"])
        dimension = len(probe[0])

        # Если дальнейшая запись прервётся, частичный индекс не будет
        # объявлен готовым.
        self.__write_manifest(ready=False, records=0)

        if collection_exists:
            await self.qdrant_client.delete_collection(COLLECTION)

        await self.qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

        batch_size = 8

        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            vectors = await self.ollama_client.embed(
                [embedding_text(record) for record in batch]
            )

            points = []
            for record, vector in zip(batch, vectors, strict=True):
                identity = f"{record.path}:{record.kind}:{record.start_line}:{record.name}"

                points.append(
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, identity)),
                        vector=vector,
                        payload=record.model_dump(mode="json"),
                    )
                )

            await self.qdrant_client.upsert(
                collection_name=COLLECTION,
                points=points,
                wait=True,
            )

            print(
                f"Embeddings: {min(offset + batch_size, len(records))}"
                f"/{len(records)}",
                flush=True,
            )

        self.__write_manifest(ready=True, records=len(records))

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Search query must not be empty")

        if not await self.exists():
            raise RuntimeError("Semantic index is absent or incomplete")

        # Instruct рекомендуется для qwen эмбединг модели, конструкция не универсальная.
        query_text = (
            "Instruct: Retrieve source code descriptions relevant to the software question.\n"
            f"Query: {query}"
        )
        vector = (await self.ollama_client.embed([query_text]))[0]

        response = await self.qdrant_client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=max(1, min(limit, 8)),
            with_payload=True,
        )

        return [SearchHit.model_validate(point.payload) for point in response.points]
