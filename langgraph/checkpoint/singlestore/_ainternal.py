"""Shared async utility functions for the SingleStore checkpoint & storage classes."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from singlestoredb.connection import Connection


@asynccontextmanager
async def get_connection(conn: Connection | Any) -> AsyncIterator[Connection]:
	"""Get a SingleStore connection from a connection object."""
	if isinstance(conn, Connection):
		yield conn
	else:
		raise TypeError(f"Invalid connection type: {type(conn)}")
