from collections.abc import AsyncIterator, Iterator

import pytest
import singlestoredb
from singlestoredb.connection import Connection

DEFAULT_SINGLESTORE_URI = "root:test_password_123@localhost:33071"
DEFAULT_URI = "root:test_password_123@localhost:33071/test_db"


@pytest.fixture(scope="function")
def conn() -> Iterator[Connection]:
    """Sync connection fixture for SingleStore."""
    with singlestoredb.connect(
        DEFAULT_URI, autocommit=True, results_type="dict"
    ) as conn:
        yield conn


@pytest.fixture(scope="function")
async def aconn() -> AsyncIterator[Connection]:
    """Async connection fixture for SingleStore."""
    with singlestoredb.connect(
        DEFAULT_URI, autocommit=True, results_type="dict"
    ) as conn:
        yield conn


@pytest.fixture(scope="function", autouse=True)
def clear_test_db(conn: Connection) -> None:
    """Delete all tables before each test."""
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM checkpoints")
            cursor.execute("DELETE FROM checkpoint_blobs")
            cursor.execute("DELETE FROM checkpoint_writes")
            cursor.execute("DELETE FROM checkpoint_migrations")
    except Exception:
        # Tables might not exist yet
        pass
