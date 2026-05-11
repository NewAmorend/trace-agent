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


def infer_task_from_source(source_path: str, include_parent: bool = False) -> str:
    """Infer a task name from a source file path.

    Args:
        source_path: File path string.
        include_parent: If True, prepend the parent directory name (Claude convention).
    """
    if not source_path:
        return 'Unknown task'
    parts = source_path.replace('\\', '/').split('/')
    stem = parts[-1].rsplit('.', 1)[0]
    if include_parent and len(parts) >= 2:
        name = f"{parts[-2]} {stem}"
    else:
        name = stem
    return name.replace('_', ' ').replace('-', ' ')
