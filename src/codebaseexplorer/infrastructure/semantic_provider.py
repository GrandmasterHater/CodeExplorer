# Literal ограничивает тип конкретными допустимыми значениями.
from typing import Literal, Protocol

from pydantic import BaseModel


# Один результат семантического поиска.
class SearchHit(BaseModel):
    kind: Literal["class", "function", "config"]
    path: str
    description: str
    name: str | None = None


class SemanticProvider(Protocol):
    async def exists(self) -> bool: ...

    async def search(self,query: str, *,limit: int,) -> list[SearchHit]: ...
