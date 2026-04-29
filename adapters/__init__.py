"""Adapter registry for trajectory format detection and conversion."""

from adapters.base import BaseAdapter
from adapters.internal import InternalAdapter
from adapters.codex_adapter import CodexAdapter

_ADAPTERS: list[BaseAdapter] = [
    InternalAdapter(),
    CodexAdapter(),
]


def get_adapter(data: object) -> BaseAdapter | None:
    """Find and return the first adapter that can handle the given data."""
    for adapter in _ADAPTERS:
        if adapter.detect(data):
            return adapter
    return None


def register_adapter(adapter: BaseAdapter) -> None:
    """Register a custom adapter. Added at the front of the detection list."""
    _ADAPTERS.insert(0, adapter)
