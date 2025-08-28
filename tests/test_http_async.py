# type: ignore
"""Comprehensive tests for HTTP-based async checkpoint operations using pytest-httpx."""

import asyncio
import base64
import json
from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.runnables import RunnableConfig
from pytest_httpx import HTTPXMock

from langgraph.checkpoint.base import (
	CheckpointMetadata,
	create_checkpoint,
	empty_checkpoint,
)
from langgraph.checkpoint.singlestore.http.aio import AsyncHTTPSingleStoreSaver
from langgraph.checkpoint.singlestore.http.client import HTTPClientError, RetryConfig
from tests.test_utils import (
	create_test_checkpoints,
	filter_checkpoints,
	create_large_metadata,
	create_unicode_metadata,
	create_empty_checkpoint,
	create_checkpoint_with_binary_data,
	create_search_test_queries,
)


@pytest.fixture
def base_url() -> str:
	"""Base URL for testing."""
	return "http://localhost:8080"


@pytest.fixture
def api_key() -> str:
	"""API key for testing."""
	return "test-api-key"


@pytest_asyncio.fixture
async def async_saver(base_url: str, api_key: str) -> AsyncHTTPSingleStoreSaver:
	"""Create AsyncHTTPSingleStoreSaver instance."""
	return AsyncHTTPSingleStoreSaver(base_url=base_url, api_key=api_key)


@pytest.fixture
def sample_checkpoint() -> dict[str, Any]:
	"""Create a sample checkpoint."""
	return {
		"v": 1,
		"id": "checkpoint-1",
		"ts": "2024-01-01T00:00:00Z",
		"channel_values": {"foo": "bar"},
		"channel_versions": {},
		"versions_seen": {},
		"pending_sends": [],
	}


@pytest.fixture
def sample_config() -> RunnableConfig:
	"""Create a sample configuration."""
	return {
		"configurable": {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
		}
	}


class TestAsyncHTTPCheckpoint:
	"""Test class for async HTTP checkpoint operations."""

	@pytest.mark.asyncio
	async def test_setup_success(self, async_saver: AsyncHTTPSingleStoreSaver, httpx_mock: HTTPXMock):
		"""Test successful database setup via HTTP."""
		httpx_mock.add_response(
			method="POST",
			url="http://localhost:8080/setup",
			json={"success": True, "version": 10, "message": "Setup complete"},
		)

		await async_saver.setup()

		# Verify the request was made with correct headers
		request = httpx_mock.get_request()
		assert request.headers["Authorization"] == "Bearer test-api-key"
		assert request.headers["Content-Type"] == "application/json"

	@pytest.mark.asyncio
	async def test_setup_failure(self, async_saver: AsyncHTTPSingleStoreSaver, httpx_mock: HTTPXMock):
		"""Test failed database setup via HTTP."""
		httpx_mock.add_response(
			method="POST",
			url="http://localhost:8080/setup",
			json={"success": False, "message": "Migration failed"},
		)

		with pytest.raises(HTTPClientError) as exc_info:
			await async_saver.setup()

		assert "Setup failed: Migration failed" in str(exc_info.value)

	@pytest.mark.asyncio
	async def test_put_and_get_checkpoint(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
		sample_config: RunnableConfig,
	):
		"""Test saving and retrieving a checkpoint via HTTP."""
		# Mock PUT response
		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		# Create checkpoint
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
		metadata: CheckpointMetadata = {"source": "test", "step": 1}

		result_config = await async_saver.aput(sample_config, checkpoint, metadata, {})

		assert result_config["configurable"]["checkpoint_id"] == checkpoint["id"]

		# Verify PUT request
		put_request = httpx_mock.get_request()
		assert put_request.method == "PUT"
		request_body = json.loads(put_request.content)
		assert request_body["thread_id"] == "thread-1"
		assert request_body["checkpoint_ns"] == ""
		assert request_body["checkpoint_id"] == checkpoint["id"]

		# Mock GET response for specific checkpoint
		httpx_mock.add_response(
			method="GET",
			url=f"http://localhost:8080/checkpoints/thread-1//latest",
			json={
				"thread_id": "thread-1",
				"checkpoint_ns": "",
				"checkpoint_id": checkpoint["id"],
				"parent_checkpoint_id": None,
				"checkpoint": checkpoint,
				"metadata": metadata,
				"channel_values": [],
				"pending_writes": [],
			},
		)

		# Get checkpoint
		checkpoint_tuple = await async_saver.aget_tuple(sample_config)

		assert checkpoint_tuple is not None
		assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]
		assert checkpoint_tuple.metadata["source"] == "test"

	@pytest.mark.asyncio
	@pytest.mark.parametrize(
		"filter_params,expected_params",
		[
			({"source": "input"}, {"metadata_filter": {"source": "input"}}),
			({"step": 1, "writes": {"foo": "bar"}}, {"metadata_filter": {"step": 1, "writes": {"foo": "bar"}}}),
			({}, {}),
		],
	)
	async def test_list_checkpoints_with_filters(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
		filter_params: dict[str, Any],
		expected_params: dict[str, Any],
	):
		"""Test listing checkpoints with various filters."""
		params = {"thread_id": "thread-1", "checkpoint_ns": "", **expected_params}
		httpx_mock.add_response(
			method="GET",
			url=httpx.URL("http://localhost:8080/checkpoints", params=params),
			json={
				"checkpoints": [
					{
						"thread_id": "thread-1",
						"checkpoint_ns": "",
						"checkpoint_id": "checkpoint-1",
						"parent_checkpoint_id": None,
						"checkpoint": {
							"v": 1,
							"id": "checkpoint-1",
							"ts": "2024-01-01T00:00:00Z",
							"channel_values": {},
							"channel_versions": {},
							"versions_seen": {},
							"pending_sends": [],
						},
						"metadata": {"source": "input", "step": 1},
						"channel_values": [],
						"pending_writes": [],
					}
				],
				"total": 1,
			},
		)

		config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

		checkpoints = [checkpoint async for checkpoint in async_saver.alist(config, filter=filter_params)]

		assert len(checkpoints) == 1
		assert checkpoints[0].checkpoint["id"] == "checkpoint-1"

	@pytest.mark.asyncio
	async def test_put_writes(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test storing checkpoint writes via HTTP."""
		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoint-writes",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": "thread-1",
				"checkpoint_ns": "",
				"checkpoint_id": "checkpoint-1",
			}
		}
		writes = [("channel1", "value1"), ("channel2", {"key": "value2"})]

		await async_saver.aput_writes(config, writes, task_id="task-1", task_path="path/1")

		# Verify the request
		request = httpx_mock.get_request()
		request_body = json.loads(request.content)
		assert request_body["thread_id"] == "thread-1"
		assert request_body["checkpoint_ns"] == ""
		assert request_body["checkpoint_id"] == "checkpoint-1"
		assert request_body["task_id"] == "task-1"
		assert request_body["task_path"] == "path/1"
		assert len(request_body["writes"]) == 2

	@pytest.mark.asyncio
	async def test_delete_thread(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test deleting a thread via HTTP."""
		httpx_mock.add_response(
			method="DELETE",
			url="http://localhost:8080/threads/thread-1",
			json={
				"success": True,
				"deleted_checkpoints": 5,
				"deleted_blobs": 10,
				"deleted_writes": 3,
			},
		)

		await async_saver.adelete_thread("thread-1")

		request = httpx_mock.get_request()
		assert request.method == "DELETE"
		assert str(request.url).endswith("/threads/thread-1")

	@pytest.mark.asyncio
	async def test_checkpoint_not_found(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling 404 error when checkpoint not found."""
		httpx_mock.add_response(
			method="GET",
			url="http://localhost:8080/checkpoints/non-existent//checkpoint-1",
			status_code=404,
			json={
				"error": "NotFound",
				"message": "Checkpoint not found",
				"status_code": 404,
			},
		)

		config = {
			"configurable": {
				"thread_id": "non-existent",
				"checkpoint_ns": "",
				"checkpoint_id": "checkpoint-1",
			}
		}

		result = await async_saver.aget_tuple(config)

		assert result is None

	@pytest.mark.asyncio
	async def test_retry_on_server_error(
		self,
		base_url: str,
		api_key: str,
		httpx_mock: HTTPXMock,
	):
		"""Test retry logic on server errors."""
		# Configure saver with custom retry config
		retry_config = RetryConfig(max_retries=2, backoff_factor=0.01)
		async_saver = AsyncHTTPSingleStoreSaver(
			base_url=base_url,
			api_key=api_key,
			retry_config=retry_config,
		)

		# First two requests fail with 500, third succeeds
		httpx_mock.add_response(
			method="GET",
			url="http://localhost:8080/checkpoints/thread-1//latest",
			status_code=500,
		)
		httpx_mock.add_response(
			method="GET",
			url="http://localhost:8080/checkpoints/thread-1//latest",
			status_code=500,
		)
		httpx_mock.add_response(
			method="GET",
			url="http://localhost:8080/checkpoints/thread-1//latest",
			json={
				"thread_id": "thread-1",
				"checkpoint_ns": "",
				"checkpoint_id": "checkpoint-1",
				"parent_checkpoint_id": None,
				"checkpoint": {
					"v": 1,
					"id": "checkpoint-1",
					"ts": "2024-01-01T00:00:00Z",
					"channel_values": {},
					"channel_versions": {},
					"versions_seen": {},
					"pending_sends": [],
				},
				"metadata": {},
				"channel_values": [],
				"pending_writes": [],
			},
		)

		config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

		result = await async_saver.aget_tuple(config)

		assert result is not None
		assert result.checkpoint["id"] == "checkpoint-1"

		# Verify all 3 requests were made
		requests = httpx_mock.get_requests()
		assert len(requests) == 3

	@pytest.mark.asyncio
	async def test_context_manager(
		self,
		base_url: str,
		api_key: str,
		httpx_mock: HTTPXMock,
	):
		"""Test using AsyncHTTPSingleStoreSaver with context manager."""
		httpx_mock.add_response(
			method="POST",
			url="http://localhost:8080/setup",
			json={"success": True},
		)

		async with AsyncHTTPSingleStoreSaver.from_url(
			base_url=base_url,
			api_key=api_key,
		) as saver:
			await saver.setup()

		request = httpx_mock.get_request()
		assert request.method == "POST"
		assert str(request.url).endswith("/setup")

	# @pytest.mark.asyncio
	# async def test_sync_bridge_methods(
	# 	self,
	# 	async_saver: AsyncHTTPSingleStoreSaver,
	# 	httpx_mock: HTTPXMock,
	# ):
	# 	"""Test sync bridge methods that delegate to async."""
	# 	# Mock for list operation
	# 	httpx_mock.add_response(
	# 		method="GET",
	# 		url="http://localhost:8080/checkpoints",
	# 		json={"checkpoints": [], "total": 0},
	# 	)

	# 	config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

	# 	async with async_saver._get_client() as client:
	# 		async_saver._client = client

	# 		# Test that sync methods work from a different thread
	# 		import threading

	# 		results = []

	# 		def run_sync():
	# 			# These should work from a different thread
	# 			checkpoints = list(async_saver.list(config))
	# 			results.append(("list", len(checkpoints)))

	# 		thread = threading.Thread(target=run_sync)
	# 		thread.start()
	# 		thread.join()

	# 		assert results == [("list", 0)]

	@pytest.mark.asyncio
	async def test_concurrent_requests(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling concurrent async requests."""
		# Mock multiple concurrent requests
		for i in range(5):
			httpx_mock.add_response(
				method="GET",
				url=f"http://localhost:8080/checkpoints/thread-{i}//latest",
				json={
					"thread_id": f"thread-{i}",
					"checkpoint_ns": "",
					"checkpoint_id": f"checkpoint-{i}",
					"parent_checkpoint_id": None,
					"checkpoint": {
						"v": 1,
						"id": f"checkpoint-{i}",
						"ts": "2024-01-01T00:00:00Z",
						"channel_values": {},
						"channel_versions": {},
						"versions_seen": {},
						"pending_sends": [],
					},
					"metadata": {},
					"channel_values": [],
					"pending_writes": [],
				},
			)

		# Run multiple requests concurrently
		configs = [{"configurable": {"thread_id": f"thread-{i}", "checkpoint_ns": ""}} for i in range(5)]

		results = await asyncio.gather(*[async_saver.aget_tuple(config) for config in configs])

		assert len(results) == 5
		for i, result in enumerate(results):
			assert result is not None
			assert result.checkpoint["id"] == f"checkpoint-{i}"

	@pytest.mark.asyncio
	async def test_blob_encoding_decoding(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test binary data encoding/decoding for blobs."""
		binary_data = b"This is binary data \x00\x01\x02"
		encoded = base64.b64encode(binary_data).decode("utf-8")

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
		checkpoint["channel_values"]["binary_channel"] = binary_data

		config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
		metadata = {"source": "test"}

		await async_saver.aput(config, checkpoint, metadata, {"binary_channel": "1"})

		request = httpx_mock.get_request()
		request_body = json.loads(request.content)

		# Verify blob data was encoded
		assert request_body["blob_data"] is not None
		assert len(request_body["blob_data"]) == 1
		blob = request_body["blob_data"][0]
		assert blob["channel"] == "binary_channel"
		assert blob["blob"] == encoded


class TestMetadataHandling:
	"""Test metadata handling scenarios."""

	@pytest.mark.asyncio
	async def test_null_character_handling(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling of null characters in metadata."""
		metadata_with_null = {"my_key": "\x00abc"}

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": "thread-null",
				"checkpoint_ns": "",
			}
		}
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		await async_saver.aput(config, checkpoint, metadata_with_null, {})

		request = httpx_mock.get_request()
		request_body = json.loads(request.content)
		# Note: JSON spec doesn't allow null characters in strings, so they get stripped
		assert request_body["metadata"]["my_key"] == "abc"

	@pytest.mark.asyncio
	async def test_unicode_metadata(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling of Unicode characters in metadata."""
		unicode_metadata = create_unicode_metadata()

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		httpx_mock.add_response(
			method="GET",
			url="http://localhost:8080/checkpoints/thread-unicode//latest",
			json={
				"thread_id": "thread-unicode",
				"checkpoint_ns": "",
				"checkpoint_id": "checkpoint-1",
				"parent_checkpoint_id": None,
				"checkpoint": create_checkpoint(empty_checkpoint(), {}, 1),
				"metadata": unicode_metadata,
				"channel_values": [],
				"pending_writes": [],
			},
		)

		config = {
			"configurable": {
				"thread_id": "thread-unicode",
				"checkpoint_ns": "",
			}
		}
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		await async_saver.aput(config, checkpoint, unicode_metadata, {})
		retrieved = await async_saver.aget_tuple(config)

		assert retrieved.metadata == unicode_metadata

	@pytest.mark.asyncio
	async def test_large_metadata(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling of large metadata payloads."""
		large_metadata = create_large_metadata(num_keys=50)

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": "thread-large",
				"checkpoint_ns": "",
			}
		}
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		await async_saver.aput(config, checkpoint, large_metadata, {})

		request = httpx_mock.get_request()
		request_body = json.loads(request.content)
		assert len(request_body["metadata"]) == 50


class TestSearchFunctionality:
	"""Test search and filtering capabilities."""

	@pytest.mark.asyncio
	async def test_search_with_multiple_filters(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test search functionality with various filter combinations."""
		test_queries = create_search_test_queries()
		all_checkpoints = create_test_checkpoints()

		for filter_query, expected_count in test_queries:
			filtered = filter_checkpoints(all_checkpoints, filter_query)

			expected_params = {}
			if filter_query:
				expected_params["metadata_filter"] = filter_query

			httpx_mock.add_response(
				method="GET",
				url=httpx.URL("http://localhost:8080/checkpoints", params=expected_params),
				json={"checkpoints": filtered[:expected_count], "total": expected_count},
			)

			results = []
			async for checkpoint in async_saver.alist(None, filter=filter_query):
				results.append(checkpoint)

			assert len(results) == expected_count

	@pytest.mark.asyncio
	async def test_cross_namespace_search(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test searching across multiple namespaces."""
		checkpoints = [
			{
				"thread_id": "thread-1",
				"checkpoint_ns": "",
				"checkpoint_id": "cp-1",
				"parent_checkpoint_id": None,
				"checkpoint": create_checkpoint(empty_checkpoint(), {}, 1),
				"metadata": {"source": "input"},
				"channel_values": [],
				"pending_writes": [],
			},
			{
				"thread_id": "thread-1",
				"checkpoint_ns": "inner",
				"checkpoint_id": "cp-2",
				"parent_checkpoint_id": None,
				"checkpoint": create_checkpoint(empty_checkpoint(), {}, 2),
				"metadata": {"source": "loop"},
				"channel_values": [],
				"pending_writes": [],
			},
		]

		httpx_mock.add_response(
			method="GET",
			url=httpx.URL("http://localhost:8080/checkpoints", params={"thread_id": "thread-1"}),
			json={"checkpoints": checkpoints, "total": 2},
		)

		config = {"configurable": {"thread_id": "thread-1"}}

		results = []
		async for checkpoint in async_saver.alist(config):
			results.append(checkpoint)

		assert len(results) == 2
		assert {r.config["configurable"]["checkpoint_ns"] for r in results} == {"", "inner"}


class TestConcurrentAccess:
	"""Test concurrent access scenarios."""

	@pytest.mark.asyncio
	async def test_concurrent_checkpoint_writes(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test concurrent writing of checkpoints."""
		num_threads = 10

		for _ in range(num_threads):
			httpx_mock.add_response(
				method="PUT",
				url="http://localhost:8080/checkpoints",
				json={},
			)

		async def write_checkpoint(thread_id: str, checkpoint_id: int):
			config = {
				"configurable": {
					"thread_id": thread_id,
					"checkpoint_ns": "",
				}
			}
			checkpoint = create_checkpoint(empty_checkpoint(), {}, checkpoint_id)
			metadata = {"source": "concurrent", "id": checkpoint_id}

			await async_saver.aput(config, checkpoint, metadata, {})

		tasks = [write_checkpoint(f"thread-{i}", i) for i in range(num_threads)]

		await asyncio.gather(*tasks)

		requests = httpx_mock.get_requests()
		assert len(requests) == num_threads

		thread_ids = set()
		for request in requests:
			body = json.loads(request.content)
			thread_ids.add(body["thread_id"])

		assert len(thread_ids) == num_threads

	@pytest.mark.asyncio
	async def test_concurrent_reads_and_writes(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test concurrent reading and writing operations."""
		# Mock responses for writes
		for _ in range(5):
			httpx_mock.add_response(
				method="PUT",
				url="http://localhost:8080/checkpoints",
				json={},
			)

		# Mock responses for reads
		for i in range(5):
			httpx_mock.add_response(
				method="GET",
				url=f"http://localhost:8080/checkpoints/thread-read-{i}//latest",
				json={
					"thread_id": f"thread-read-{i}",
					"checkpoint_ns": "",
					"checkpoint_id": f"checkpoint-{i}",
					"parent_checkpoint_id": None,
					"checkpoint": create_checkpoint(empty_checkpoint(), {}, i),
					"metadata": {"id": i},
					"channel_values": [],
					"pending_writes": [],
				},
			)

		async def write_operation(thread_id: str, checkpoint_id: int):
			config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
			checkpoint = create_checkpoint(empty_checkpoint(), {}, checkpoint_id)
			await async_saver.aput(config, checkpoint, {"id": checkpoint_id}, {})

		async def read_operation(thread_id: str):
			config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
			return await async_saver.aget_tuple(config)

		# Mix reads and writes
		write_tasks = [write_operation(f"thread-write-{i}", i) for i in range(5)]
		read_tasks = [read_operation(f"thread-read-{i}") for i in range(5)]

		# Execute all tasks
		await asyncio.gather(*write_tasks)
		read_results = await asyncio.gather(*read_tasks)

		# Check that reads returned expected results
		assert len(read_results) == 5

		for i, result in enumerate(read_results):
			assert result is not None
			assert result.metadata["id"] == i


class TestErrorHandling:
	"""Test error handling and edge cases."""

	@pytest.mark.asyncio
	async def test_empty_checkpoint_handling(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling of checkpoints with empty values."""
		empty_cp = create_empty_checkpoint()

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": "thread-empty",
				"checkpoint_ns": "",
			}
		}

		result = await async_saver.aput(config, empty_cp, {}, {})

		assert result["configurable"]["checkpoint_id"] == empty_cp["id"]

	@pytest.mark.asyncio
	async def test_binary_data_in_checkpoint(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling of binary data in checkpoint blobs."""
		checkpoint, binary_data = create_checkpoint_with_binary_data()
		encoded = base64.b64encode(binary_data).decode("utf-8")

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoints",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": "thread-binary",
				"checkpoint_ns": "",
			}
		}

		await async_saver.aput(config, checkpoint, {}, {"binary_channel": "1"})

		request = httpx_mock.get_request()
		request_body = json.loads(request.content)

		assert request_body["blob_data"] is not None
		assert len(request_body["blob_data"]) == 1
		blob = request_body["blob_data"][0]
		assert blob["channel"] == "binary_channel"
		assert blob["blob"] == encoded

	@pytest.mark.asyncio
	async def test_connection_error_handling(
		self,
		api_key: str,
	):
		"""Test handling of connection errors."""
		async_saver = AsyncHTTPSingleStoreSaver(
			base_url="http://non-existent-host:9999",
			api_key=api_key,
			retry_config=RetryConfig(max_retries=1, backoff_factor=0.01),
		)

		config = {"configurable": {"thread_id": "test", "checkpoint_ns": ""}}

		with pytest.raises(httpx.ConnectError):
			await async_saver.aget_tuple(config)

	@pytest.mark.asyncio
	async def test_malformed_response_handling(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test handling of malformed server responses."""
		httpx_mock.add_response(
			method="GET",
			url="http://localhost:8080/checkpoints/test//latest",
			content=b"not json",
			headers={"Content-Type": "application/json"},
		)

		config = {"configurable": {"thread_id": "test", "checkpoint_ns": ""}}

		with pytest.raises(json.JSONDecodeError):
			await async_saver.aget_tuple(config)


class TestPendingWrites:
	"""Test pending writes functionality."""

	@pytest.mark.asyncio
	async def test_multiple_pending_writes(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test storing multiple pending writes."""
		for _ in range(3):
			httpx_mock.add_response(
				method="PUT",
				url="http://localhost:8080/checkpoint-writes",
				json={},
			)

		config = {
			"configurable": {
				"thread_id": "thread-writes",
				"checkpoint_ns": "",
				"checkpoint_id": "checkpoint-1",
			}
		}

		writes = [
			[("channel1", "value1"), ("channel2", "value2")],
			[("channel3", {"nested": "object"})],
			[("channel4", b"binary data")],
		]

		for i, write_batch in enumerate(writes):
			await async_saver.aput_writes(config, write_batch, task_id=f"task-{i}")

		requests = httpx_mock.get_requests()
		assert len(requests) == 3

		for i, request in enumerate(requests):
			body = json.loads(request.content)
			assert body["task_id"] == f"task-{i}"

	@pytest.mark.asyncio
	async def test_writes_with_binary_data(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
		httpx_mock: HTTPXMock,
	):
		"""Test pending writes containing binary data."""
		binary_data = b"\x00\x01\x02\x03\x04"
		encoded = base64.b64encode(binary_data).decode("utf-8")

		httpx_mock.add_response(
			method="PUT",
			url="http://localhost:8080/checkpoint-writes",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": "thread-binary-writes",
				"checkpoint_ns": "",
				"checkpoint_id": "checkpoint-1",
			}
		}

		writes = [("binary_channel", binary_data)]

		await async_saver.aput_writes(config, writes, task_id="binary-task")

		request = httpx_mock.get_request()
		body = json.loads(request.content)
		assert len(body["writes"]) == 1
		write = body["writes"][0]
		assert write["channel"] == "binary_channel"
		assert write["blob"] == encoded

	@pytest.mark.asyncio
	async def test_event_loop_error_from_async_context(
		self,
		async_saver: AsyncHTTPSingleStoreSaver,
	):
		"""Test that sync methods raise error when called from async context."""
		config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

		# Calling sync method from async context should raise error
		with pytest.raises(asyncio.InvalidStateError) as exc_info:
			async_saver.get_tuple(config)

			assert "Synchronous calls to AsyncHTTPSingleStoreSaver" in str(exc_info.value)
