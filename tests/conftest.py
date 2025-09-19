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


@pytest.fixture(scope="session")
def setup_test_database(request):
	"""Create test database once for the entire test session.

	Only runs for tests that require database access (not marked with 'no_db').
	"""
	# Skip database setup for HTTP tests
	if request.node.get_closest_marker("no_db"):
		return

	# Create unique test database
	with singlestoredb.connect(DEFAULT_URI_WITHOUT_DB, autocommit=True, results_type="dict") as conn:
		with conn.cursor() as cursor:
			cursor.execute(f"CREATE DATABASE IF NOT EXISTS {TEST_DB_NAME}")

	yield

	# Clean up test database after all tests
	if not request.node.get_closest_marker("no_db"):
		with singlestoredb.connect(DEFAULT_URI_WITHOUT_DB, autocommit=True, results_type="dict") as conn:
			with conn.cursor() as cursor:
				cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")


@pytest.fixture(scope="class")
def conn(request, setup_test_database) -> Iterator[Connection]:
	"""Class-scoped sync connection fixture for SingleStore.

	Skips connection for tests marked with 'no_db'.
	"""
	if request.node.get_closest_marker("no_db"):
		yield None
		return

	with singlestoredb.connect(DEFAULT_URI_WITH_DB, autocommit=True, results_type="dict") as conn:
		yield conn


@pytest.fixture(scope="class")
async def aconn(request, setup_test_database) -> AsyncIterator[Connection]:
	"""Class-scoped async connection fixture for SingleStore.

	Skips connection for tests marked with 'no_db'.
	"""
	if request.node.get_closest_marker("no_db"):
		yield None
		return

	# SingleStore doesn't support async context managers, so we use sync connection
	with singlestoredb.connect(DEFAULT_URI_WITH_DB, autocommit=True, results_type="dict") as conn:
		yield conn


@pytest.fixture(scope="class", autouse=True)
def clear_test_db(request) -> None:
	"""Delete all tables before each test class.

	Skips for tests marked with 'no_db'.
	"""
	if request.node.get_closest_marker("no_db"):
		return

	# Only access conn if not marked with no_db
	conn = request.getfixturevalue("conn")
	if conn is None:
		return

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
def clear_test_data(request) -> None:
	"""Clear test data before each test to ensure isolation.

	Skips for tests marked with 'no_db'.
	"""
	if request.node.get_closest_marker("no_db"):
		return

	# Only access conn if not marked with no_db
	conn = request.getfixturevalue("conn")
	if conn is None:
		return

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


# ==================== HTTP Test Mode Configuration ====================


def pytest_addoption(parser):
	"""Add command line options for HTTP testing modes."""
	parser.addoption(
		"--use-real-server", action="store_true", default=False, help="Run HTTP tests with real server instead of mocks"
	)
	parser.addoption(
		"--server-url",
		action="store",
		default="http://localhost:8080",
		help="URL of the HTTP server (for real server mode)",
	)
	parser.addoption(
		"--base-path",
		action="store",
		default="",
		help="Base path for the API endpoints (e.g., '/api/v1')",
	)
	parser.addoption("--api-key", action="store", default=None, help="API key for the HTTP server")


def pytest_configure(config):
	"""Configure pytest with custom markers."""
	config.addinivalue_line("markers", "mock_only: Run test only with mocked responses (skip when --use-real-server)")
	config.addinivalue_line(
		"markers", "real_server_only: Run test only with real server (skip when not --use-real-server)"
	)


def pytest_collection_modifyitems(config, items):
	"""Skip tests based on markers and mode."""
	use_real_server = config.getoption("--use-real-server")

	for item in items:
		# Skip mock-only tests when using real server
		if use_real_server and item.get_closest_marker("mock_only"):
			skip_mark = pytest.mark.skip(reason="Test requires mock mode")
			item.add_marker(skip_mark)

		# Skip real-server-only tests when using mocks
		if not use_real_server and item.get_closest_marker("real_server_only"):
			skip_mark = pytest.mark.skip(reason="Test requires real server")
			item.add_marker(skip_mark)


@pytest.fixture
def http_test_mode(request) -> bool:
	"""Check if we're running with real server."""
	return request.config.getoption("--use-real-server")


@pytest.fixture
def http_base_url(request, http_test_mode) -> str:
	"""Get base URL for HTTP tests, including base path if provided."""
	if http_test_mode:
		server_url = request.config.getoption("--server-url")
		base_path = request.config.getoption("--base-path")
		# Remove trailing slash from server URL and leading slash from base path
		server_url = server_url.rstrip("/")
		if base_path:
			base_path = "/" + base_path.strip("/")
			return server_url + base_path
		return server_url
	return "http://localhost:8080"  # Default for mocks


@pytest.fixture
def http_api_key(request, http_test_mode) -> str | None:
	"""Get API key for HTTP tests."""
	if http_test_mode:
		return request.config.getoption("--api-key")
	return "test-api-key"  # Default for mocks


@pytest.fixture
def httpx_mock_if_enabled(request, http_test_mode):
	"""Provide httpx_mock only when not using real server."""
	if not http_test_mode:
		# Use the built-in httpx_mock fixture from pytest-httpx
		return request.getfixturevalue("httpx_mock")
	return None
