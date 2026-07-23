"""
Helper functions.
"""

from pathlib import Path


def create_directory(path: Path) -> None:
    """
    Create a directory if it does not exist.
    """

    path.mkdir(parents=True, exist_ok=True)