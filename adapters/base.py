"""Abstract base class for trajectory format adapters."""

from abc import ABC, abstractmethod
from models import Trajectory


class BaseAdapter(ABC):
    """Base adapter for converting trajectory data to unified Step format."""

    @abstractmethod
    def detect(self, data: object) -> bool:
        """Return True if this adapter can handle the given data."""

    @abstractmethod
    def transform(self, data: object, source_path: str = "") -> Trajectory:
        """Convert data to a Trajectory."""
