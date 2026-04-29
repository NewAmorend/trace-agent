"""Abstract base class for trajectory format adapters."""

from abc import ABC, abstractmethod
from models import Step


class BaseAdapter(ABC):
    """Base adapter for converting trajectory data to unified Step format."""

    @abstractmethod
    def detect(self, data: object) -> bool:
        """Return True if this adapter can handle the given data."""

    @abstractmethod
    def transform(self, data: object) -> tuple[str, str, list[Step]]:
        """Convert data to (task, final_status, steps)."""
