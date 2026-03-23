"""Decorator namespace helpers for developer_sdk."""

from .constants import Constant


class CrowdioNamespace:
    """Namespace class for @crowdio.task decorator style."""

    Constant = Constant

    @staticmethod
    def task(*args, **kwargs):
        # Import lazily to avoid circular imports with decorators.py.
        from .decorators import task

        return task(*args, **kwargs)


crowdio = CrowdioNamespace()


__all__ = ["CrowdioNamespace", "crowdio"]
