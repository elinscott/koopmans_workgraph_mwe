from typing import Protocol, runtime_checkable


@runtime_checkable
class HasKind(Protocol):
    kind: str


