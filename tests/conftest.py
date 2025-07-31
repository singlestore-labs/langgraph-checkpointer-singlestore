from collections.abc import AsyncIterator, Iterator
import uuid
from typing import Any

import pytest
import singlestoredb
from singlestoredb.connection import Connection
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
	Checkpoint,
	CheckpointMetadata,
	create_checkpoint,
	empty_checkpoint,
)

DEFAULT_URI_WITHOUT_DB = "root:test@127.0.0.1:33071"
TEST_DB_NAME = f"test_db_{uuid.uuid4().hex[:8]}"
DEFAULT_URI_WITH_DB = f"root:test@127.0.0.1:33071/{TEST_DB_NAME}"


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
	"""Create test database once for the entire test session."""
	# Create unique test database
	with singlestoredb.connect(DEFAULT_URI_WITHOUT_DB, autocommit=True, results_type="dict") as conn:
		with conn.cursor() as cursor:
			cursor.execute(f"CREATE DATABASE IF NOT EXISTS {TEST_DB_NAME}")

	yield

	# Clean up test database after all tests
	with singlestoredb.connect(DEFAULT_URI_WITHOUT_DB, autocommit=True, results_type="dict") as conn:
		with conn.cursor() as cursor:
			cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


@pytest.fixture(scope="class")
def conn() -> Iterator[Connection]:
	"""Class-scoped sync connection fixture for SingleStore."""
	with singlestoredb.connect(DEFAULT_URI_WITH_DB, autocommit=True, results_type="dict") as conn:
		yield conn


@pytest.fixture(scope="class")
async def aconn() -> AsyncIterator[Connection]:
	"""Class-scoped async connection fixture for SingleStore."""
	# SingleStore doesn't support async context managers, so we use sync connection
	with singlestoredb.connect(DEFAULT_URI_WITH_DB, autocommit=True, results_type="dict") as conn:
		yield conn


@pytest.fixture(scope="class", autouse=True)
def clear_test_db(conn: Connection) -> None:
	"""Delete all tables before each test class."""
	try:
		with conn.cursor() as cursor:
			cursor.execute("DELETE FROM checkpoints")
			cursor.execute("DELETE FROM checkpoint_blobs")
			cursor.execute("DELETE FROM checkpoint_writes")
			cursor.execute("DELETE FROM checkpoint_migrations")
	except Exception:
		# Tables might not exist yet
		pass


@pytest.fixture(scope="function", autouse=True)
def clear_test_data(conn: Connection) -> None:
	"""Clear test data before each test to ensure isolation."""
	try:
		with conn.cursor() as cursor:
			cursor.execute("DELETE FROM checkpoints")
			cursor.execute("DELETE FROM checkpoint_blobs")
			cursor.execute("DELETE FROM checkpoint_writes")
			cursor.execute("DELETE FROM checkpoint_migrations")
	except Exception:
		# Tables might not exist yet
		pass


@pytest.fixture(scope="class")
def sync_saver(conn: Connection):
	"""Class-scoped sync saver fixture."""
	from langgraph.checkpoint.singlestore import SingleStoreSaver

	saver = SingleStoreSaver(conn)
	try:
		saver.setup()
	except Exception as e:
		# Ignore duplicate index errors since we're reusing the database
		if "Duplicate key name" not in str(e):
			raise
	return saver


@pytest.fixture(scope="class")
async def async_saver(aconn: Connection):
	"""Class-scoped async saver fixture."""
	from langgraph.checkpoint.singlestore.aio import AsyncSingleStoreSaver

	saver = AsyncSingleStoreSaver(aconn)
	try:
		await saver.setup()
	except Exception as e:
		# Ignore duplicate index errors since we're reusing the database
		if "Duplicate key name" not in str(e):
			raise
	return saver


@pytest.fixture(scope="function")
def test_data():
	"""Function-scoped fixture providing test data for checkpoint tests."""
	import uuid

	# Generate unique identifiers to prevent test conflicts
	unique_id = uuid.uuid4().hex[:8]

	config_1: RunnableConfig = {
		"configurable": {
			"thread_id": f"thread-1-{unique_id}",
			"checkpoint_id": "1",
			"checkpoint_ns": "",
		}
	}
	config_2: RunnableConfig = {
		"configurable": {
			"thread_id": f"thread-2-{unique_id}",
			"checkpoint_id": "2",
			"checkpoint_ns": "",
		}
	}
	config_3: RunnableConfig = {
		"configurable": {
			"thread_id": f"thread-2-{unique_id}",
			"checkpoint_id": "2-inner",
			"checkpoint_ns": "inner",
		}
	}

	chkpnt_1: Checkpoint = empty_checkpoint()
	chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
	chkpnt_3: Checkpoint = empty_checkpoint()

	metadata_1: CheckpointMetadata = {
		"source": "input",
		"step": 2,
		"writes": {},
		"score": 1,
	}
	metadata_2: CheckpointMetadata = {
		"source": "loop",
		"step": 1,
		"writes": {"foo": "bar"},
		"score": None,
	}
	metadata_3: CheckpointMetadata = {}

	return {
		"configs": [config_1, config_2, config_3],
		"checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
		"metadata": [metadata_1, metadata_2, metadata_3],
	}
