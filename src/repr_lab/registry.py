from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple named registry for research components."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        if name in self._items:
            msg = f"{self.kind} '{name}' is already registered"
            raise KeyError(msg)
        self._items[name] = item
        return item

    def decorator(self, name: str) -> Callable[[T], T]:
        def _register(item: T) -> T:
            return self.register(name, item)

        return _register

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<empty>"
            msg = f"Unknown {self.kind} '{name}'. Known values: {known}"
            raise KeyError(msg) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, T]]:
        return [(name, self._items[name]) for name in self.names()]

    def as_dict(self) -> dict[str, T]:
        return {name: self._items[name] for name in self.names()}
