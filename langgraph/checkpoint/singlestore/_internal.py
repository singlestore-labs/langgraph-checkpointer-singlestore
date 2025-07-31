"""Shared utility functions for the SingleStore checkpoint & storage classes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from singlestoredb.connection import Connection


@contextmanager
def get_connection(conn: Connection | Any) -> Iterator[Connection]:
    """Get a SingleStore connection from a connection object."""
    if isinstance(conn, Connection):
        yield conn
    else:
        raise TypeError(f"Invalid connection type: {type(conn)}")