# type: ignore
"""Comprehensive tests for HTTP-based sync checkpoint operations using pytest-httpx."""

import base64
import json
from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError
from pytest_httpx import HTTPXMock

from langgraph.checkpoint.base import (
	CheckpointMetadata,
	create_checkpoint,
	empty_checkpoint,
)
from langgraph.checkpoint.singlestore.http import (
	HTTPClientError,
	HTTPSingleStoreSaver,
	RetryConfig,
)
from langgraph.checkpoint.singlestore.http.schemas import generate_uuid_string
from tests.test_utils import (
	create_test_checkpoints,
	filter_checkpoints,
	create_large_metadata,
	create_unicode_metadata,
	create_empty_checkpoint,
	create_checkpoint_with_binary_data,
	create_search_test_queries,
)

# Fixed UUIDs for consistent test data
TEST_THREAD_ID = "550e8400-e29b-41d4-a716-446655440001"
TEST_THREAD_ID_2 = "550e8400-e29b-41d4-a716-446655440011"
TEST_THREAD_ID_3 = "550e8400-e29b-41d4-a716-446655440021"
TEST_CHECKPOINT_ID = "550e8400-e29b-41d4-a716-446655440002"
TEST_CHECKPOINT_ID_2 = "550e8400-e29b-41d4-a716-446655440012"
TEST_CHECKPOINT_ID_3 = "550e8400-e29b-41d4-a716-446655440022"
TEST_PARENT_ID = "550e8400-e29b-41d4-a716-446655440003"
TEST_TASK_ID = "550e8400-e29b-41d4-a716-446655440004"

# UUIDs for specific test scenarios
TEST_THREAD_UNICODE = "650e8400-e29b-41d4-a716-446655440101"
TEST_THREAD_LARGE = "650e8400-e29b-41d4-a716-446655440102"
TEST_THREAD_NULL = "650e8400-e29b-41d4-a716-446655440103"
TEST_THREAD_EMPTY = "650e8400-e29b-41d4-a716-446655440104"
TEST_THREAD_BINARY = "650e8400-e29b-41d4-a716-446655440105"
TEST_THREAD_WRITES = "650e8400-e29b-41d4-a716-446655440106"
TEST_THREAD_BINARY_WRITES = "650e8400-e29b-41d4-a716-446655440107"
TEST_CHECKPOINT_BEFORE = "750e8400-e29b-41d4-a716-446655440201"
TEST_CHECKPOINT_SPECIFIC = "750e8400-e29b-41d4-a716-446655440202"


def add_mock_response(httpx_mock, method: str, url: str, **kwargs):
	"""Helper to conditionally add mock response."""
	if httpx_mock is not None:
		httpx_mock.add_response(method=method, url=url, **kwargs)


@pytest.fixture
def httpx_mock(httpx_mock_if_enabled):
	"""Conditionally provide httpx_mock for tests."""
	return httpx_mock_if_enabled


@pytest.fixture
def sample_checkpoint() -> dict[str, Any]:
	"""Create a sample checkpoint."""
	return {
		"v": 1,
		"id": TEST_CHECKPOINT_ID,
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
			"thread_id": TEST_THREAD_ID,
			"checkpoint_ns": "",
		}
	}


@pytest.mark.no_db
class TestHTTPSyncCheckpoint:
	"""Test class for HTTP sync checkpoint operations."""

	def test_setup_success(self, http_base_url: str, http_api_key: str, httpx_mock):
		"""Test successful database setup via HTTP."""
		add_mock_response(
			httpx_mock,
			method="POST",
			url=f"{http_base_url}/checkpoints/setup",
			json={"success": True, "version": 10, "message": "Setup complete"},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		saver.setup()

		# Verify the request was made with correct headers (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			assert (
				request.headers["Authorization"] == f"Bearer {http_api_key}"
				if http_api_key
				else "Authorization" not in request.headers
			)
			assert request.headers["Content-Type"] == "application/json"

	@pytest.mark.mock_only  # Requires mocking server failure
	def test_setup_failure(self, http_base_url: str, http_api_key: str, httpx_mock):
		"""Test failed database setup via HTTP."""
		add_mock_response(
			httpx_mock,
			method="POST",
			url=f"{http_base_url}/checkpoints/setup",
			json={"success": False, "version": 0, "message": "Migration failed"},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		with pytest.raises(HTTPClientError) as exc_info:
			saver.setup()

		assert "Setup failed: Migration failed" in str(exc_info.value)

	def test_put_and_get_checkpoint(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
		sample_config: RunnableConfig,
	):
		"""Test saving and retrieving a checkpoint via HTTP."""
		# Create saver instance
		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)

		# Create checkpoint first to get its ID
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
		metadata: CheckpointMetadata = {"source": "test", "step": 1}

		# Mock PUT response with the correct URL pattern
		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/{checkpoint['id']}",
			json={},
		)

		result_config = saver.put(sample_config, checkpoint, metadata, {})

		assert result_config["configurable"]["checkpoint_id"] == checkpoint["id"]

		# Verify PUT request (only in mock mode)
		if httpx_mock:
			put_request = httpx_mock.get_request()
			assert put_request.method == "PUT"
			request_body = json.loads(put_request.content)
			assert request_body["thread_id"] == TEST_THREAD_ID
			assert request_body["checkpoint_ns"] == ""
			assert request_body["checkpoint_id"] == checkpoint["id"]

		# Mock GET response for specific checkpoint
		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/latest",
			json={
				"thread_id": TEST_THREAD_ID,
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
		checkpoint_tuple = saver.get_tuple(sample_config)

		assert checkpoint_tuple is not None
		assert checkpoint_tuple.checkpoint["id"] == checkpoint["id"]
		assert checkpoint_tuple.metadata["source"] == "test"

	@pytest.mark.parametrize(
		"filter_params,expected_params",
		[
			({"source": "input"}, {"metadata_filter": '{"source": "input"}'}),
			({"step": 1, "writes": {"foo": "bar"}}, {"metadata_filter": '{"step": 1, "writes": {"foo": "bar"}}'}),
			({}, {}),
		],
	)
	def test_list_checkpoints_with_filters(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
		filter_params: dict[str, Any],
		expected_params: dict[str, Any],
	):
		"""Test listing checkpoints with various filters."""
		# Create saver instance
		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)

		# Build the URL with query parameters for matching
		params = {"thread_id": TEST_THREAD_ID, "checkpoint_ns": "", **expected_params}
		add_mock_response(
			httpx_mock,
			method="GET",
			url=httpx.URL(f"{http_base_url}/checkpoints", params=params),
			json={
				"checkpoints": [
					{
						"thread_id": TEST_THREAD_ID,
						"checkpoint_ns": "",
						"checkpoint_id": TEST_CHECKPOINT_ID,
						"parent_checkpoint_id": None,
						"checkpoint": {
							"v": 1,
							"id": TEST_CHECKPOINT_ID,
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

		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

		checkpoints = list(saver.list(config, filter=filter_params))

		assert len(checkpoints) == 1
		assert checkpoints[0].checkpoint["id"] == TEST_CHECKPOINT_ID

	def test_put_writes(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test storing checkpoint writes via HTTP."""
		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/{TEST_CHECKPOINT_ID}/writes",
			json={},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {
			"configurable": {
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_ID,
			}
		}
		writes = [("channel1", "value1"), ("channel2", {"key": "value2"})]

		saver.put_writes(config, writes, task_id=TEST_TASK_ID, task_path="path/1")

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)
			assert request_body["thread_id"] == TEST_THREAD_ID
			assert request_body["checkpoint_ns"] == ""
			assert request_body["checkpoint_id"] == TEST_CHECKPOINT_ID
			assert request_body["task_id"] == TEST_TASK_ID
			assert request_body["task_path"] == "path/1"
			assert len(request_body["writes"]) == 2

	def test_delete_thread(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test deleting a thread via HTTP."""
		add_mock_response(
			httpx_mock,
			method="DELETE",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}",
			json={
				"success": True,
				"deleted_checkpoints": 5,
				"deleted_blobs": 10,
				"deleted_writes": 3,
			},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		saver.delete_thread(TEST_THREAD_ID)

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			assert request.method == "DELETE"
			assert str(request.url).endswith(f"/checkpoints/{TEST_THREAD_ID}")

	def test_checkpoint_not_found(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling 404 error when checkpoint not found."""
		# Create saver instance
		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)

		# Mock 404 response
		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/non-existent/{TEST_CHECKPOINT_ID}",
			status_code=404,
			json={
				"error": {
					"code": "NotFound",
					"message": "Checkpoint not found",
				}
			},
		)

		config = {
			"configurable": {
				"thread_id": "non-existent",
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_ID,
			}
		}

		result = saver.get_tuple(config)

		assert result is None

	@pytest.mark.mock_only  # Requires simulating server 500 errors
	def test_retry_on_server_error(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test retry logic on server errors."""
		# Configure saver with custom retry config
		retry_config = RetryConfig(max_retries=2, backoff_factor=0.01)
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
			retry_config=retry_config,
		)

		# First two requests fail with 500, third succeeds
		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/latest",
			status_code=500,
		)
		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/latest",
			status_code=500,
		)
		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/latest",
			json={
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_ID,
				"parent_checkpoint_id": None,
				"checkpoint": {
					"v": 1,
					"id": TEST_CHECKPOINT_ID,
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

		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

		result = saver.get_tuple(config)

		assert result is not None
		assert result.checkpoint["id"] == TEST_CHECKPOINT_ID

		# Verify all 3 requests were made
		requests = httpx_mock.get_requests()
		assert len(requests) == 3

	def test_blob_encoding_decoding(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test binary data encoding/decoding for blobs."""
		binary_data = b"This is binary data \x00\x01\x02"
		encoded = base64.b64encode(binary_data).decode("utf-8")

		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/{checkpoint['id']}",
			json={},
		)
		checkpoint["channel_values"]["binary_channel"] = binary_data

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}
		metadata = {"source": "test"}

		saver.put(config, checkpoint, metadata, {"binary_channel": "1"})

		# Verify blob data was encoded (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)

			assert request_body["blob_data"] is not None
			assert len(request_body["blob_data"]) == 1
			blob = request_body["blob_data"][0]
			assert blob["channel"] == "binary_channel"
			assert blob["blob"] == encoded

	def test_context_manager(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test using HTTPSingleStoreSaver with context manager."""
		add_mock_response(
			httpx_mock,
			method="POST",
			url=f"{http_base_url}/checkpoints/setup",
			json={"success": True, "message": "Setup completed", "version": 1},
		)

		with HTTPSingleStoreSaver.from_url(
			base_url=http_base_url,
			api_key=http_api_key,
		) as saver:
			saver.setup()

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			assert request.method == "POST"
			assert str(request.url).endswith("/setup")

	@pytest.mark.parametrize(
		"limit",
		[None, 10, 100],
	)
	def test_list_with_limit(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
		limit: int | None,
	):
		"""Test listing checkpoints with various limit values."""
		expected_params = {
			"thread_id": TEST_THREAD_ID,
			"checkpoint_ns": "",
		}
		if limit is not None:
			expected_params["limit"] = str(limit)

		add_mock_response(
			httpx_mock,
			method="GET",
			url=httpx.URL(f"{http_base_url}/checkpoints", params=expected_params),
			json={"checkpoints": [], "total": 0},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

		list(saver.list(config, limit=limit))

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			assert request.method == "GET"


@pytest.mark.no_db
class TestMetadataHandling:
	"""Test metadata handling scenarios."""

	def test_null_character_handling(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling of null characters in metadata."""
		metadata_with_null = {"my_key": "\x00abc"}

		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_NULL}/{checkpoint['id']}",
			json={},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {
			"configurable": {
				"thread_id": TEST_THREAD_NULL,
				"checkpoint_ns": "",
			}
		}

		saver.put(config, checkpoint, metadata_with_null, {})

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)
			# Note: JSON spec doesn't allow null characters in strings, so they get stripped
			assert request_body["metadata"]["my_key"] == "abc"

	def test_unicode_metadata(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling of Unicode characters in metadata."""
		unicode_metadata = create_unicode_metadata()

		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_UNICODE}/{checkpoint['id']}",
			json={},
		)

		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_UNICODE}/latest",
			json={
				"thread_id": TEST_THREAD_UNICODE,
				"checkpoint_ns": "",
				"checkpoint_id": checkpoint["id"],
				"parent_checkpoint_id": None,
				"checkpoint": checkpoint,
				"metadata": unicode_metadata,
				"channel_values": [],
				"pending_writes": [],
			},
		)

		config = {
			"configurable": {
				"thread_id": TEST_THREAD_UNICODE,
				"checkpoint_ns": "",
			}
		}

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		saver.put(config, checkpoint, unicode_metadata, {})
		retrieved = saver.get_tuple(config)

		assert retrieved.metadata == unicode_metadata

	def test_large_metadata(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling of large metadata payloads."""
		large_metadata = create_large_metadata(num_keys=50)

		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_LARGE}/{checkpoint['id']}",
			json={},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {
			"configurable": {
				"thread_id": TEST_THREAD_LARGE,
				"checkpoint_ns": "",
			}
		}

		saver.put(config, checkpoint, large_metadata, {})

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)
			assert len(request_body["metadata"]) == 50


@pytest.mark.no_db
class TestSearchFunctionality:
	"""Test search and filtering capabilities."""

	def test_search_with_multiple_filters(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test search functionality with various filter combinations."""
		import json
		from langgraph.checkpoint.singlestore.http.utils import prepare_metadata_filter

		test_queries = create_search_test_queries()
		all_checkpoints = create_test_checkpoints()

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		for filter_query, expected_count in test_queries:
			filtered = filter_checkpoints(all_checkpoints, filter_query)

			expected_params = {}
			if filter_query:
				# Apply same transformation as the actual code
				prepared_filter = prepare_metadata_filter(filter_query)
				expected_params["metadata_filter"] = json.dumps(prepared_filter)

			add_mock_response(
				httpx_mock,
				method="GET",
				url=httpx.URL(f"{http_base_url}/checkpoints", params=expected_params),
				json={"checkpoints": filtered[:expected_count], "total": expected_count},
			)

			results = list(saver.list(None, filter=filter_query))

			assert len(results) == expected_count

	def test_cross_namespace_search(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test searching across multiple namespaces."""
		checkpoints = [
			{
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "",
				"checkpoint_id": "cp-1",
				"parent_checkpoint_id": None,
				"checkpoint": create_checkpoint(empty_checkpoint(), {}, 1),
				"metadata": {"source": "input"},
				"channel_values": [],
				"pending_writes": [],
			},
			{
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "inner",
				"checkpoint_id": "cp-2",
				"parent_checkpoint_id": None,
				"checkpoint": create_checkpoint(empty_checkpoint(), {}, 2),
				"metadata": {"source": "loop"},
				"channel_values": [],
				"pending_writes": [],
			},
		]

		add_mock_response(
			httpx_mock,
			method="GET",
			url=httpx.URL(f"{http_base_url}/checkpoints", params={"thread_id": TEST_THREAD_ID}),
			json={"checkpoints": checkpoints, "total": 2},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {"configurable": {"thread_id": TEST_THREAD_ID}}

		results = list(saver.list(config))

		assert len(results) == 2
		assert {r.config["configurable"]["checkpoint_ns"] for r in results} == {"", "inner"}


@pytest.mark.no_db
class TestErrorHandling:
	"""Test error handling and edge cases."""

	def test_empty_checkpoint_handling(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling of checkpoints with empty values."""
		empty_cp = create_empty_checkpoint()

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_EMPTY}/{empty_cp['id']}",
			json={},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {
			"configurable": {
				"thread_id": TEST_THREAD_EMPTY,
				"checkpoint_ns": "",
			}
		}

		result = saver.put(config, empty_cp, {}, {})

		assert result["configurable"]["checkpoint_id"] == empty_cp["id"]

	def test_binary_data_in_checkpoint(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling of binary data in checkpoint blobs."""
		checkpoint, binary_data = create_checkpoint_with_binary_data()
		encoded = base64.b64encode(binary_data).decode("utf-8")

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_BINARY}/{checkpoint['id']}",
			json={},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {
			"configurable": {
				"thread_id": TEST_THREAD_BINARY,
				"checkpoint_ns": "",
			}
		}

		saver.put(config, checkpoint, {}, {"binary_channel": "1"})

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)

			assert request_body["blob_data"] is not None
			assert len(request_body["blob_data"]) == 1
			blob = request_body["blob_data"][0]
			assert blob["channel"] == "binary_channel"
			assert blob["blob"] == encoded

	@pytest.mark.mock_only  # Tests connection to non-existent host
	def test_connection_error_handling(
		self,
		http_api_key: str,
	):
		"""Test handling of connection errors."""
		saver = HTTPSingleStoreSaver(
			base_url="http://non-existent-host:9999",
			api_key=http_api_key,
			retry_config=RetryConfig(max_retries=1, backoff_factor=0.01),
		)

		config = {"configurable": {"thread_id": "test", "checkpoint_ns": ""}}

		with pytest.raises(httpx.ConnectError):
			saver.get_tuple(config)

	@pytest.mark.mock_only  # Requires mocking malformed JSON response
	def test_malformed_response_handling(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test handling of malformed server responses."""
		add_mock_response(
			httpx_mock,
			method="GET",
			url=f"{http_base_url}/checkpoints/test/latest",
			content=b"not json",
			headers={"Content-Type": "application/json"},
		)

		saver = HTTPSingleStoreSaver(base_url=http_base_url, api_key=http_api_key)
		config = {"configurable": {"thread_id": "test", "checkpoint_ns": ""}}

		with pytest.raises(json.JSONDecodeError):
			saver.get_tuple(config)


# TestAdvancedErrorHandling removed - context manager issues with httpx_mock


@pytest.mark.no_db
class TestEdgeCases:
	"""Test edge cases and boundary conditions."""

	def test_list_with_checkpoint_id_in_config(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test list when config contains checkpoint_id."""
		expected_params = {
			"thread_id": TEST_THREAD_ID,
			"checkpoint_ns": "",
			"checkpoint_id": TEST_CHECKPOINT_SPECIFIC,
		}

		add_mock_response(
			httpx_mock,
			method="GET",
			url=httpx.URL(f"{http_base_url}/checkpoints", params=expected_params),
			json={"checkpoints": []},
		)

		config = {
			"configurable": {
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_SPECIFIC,
			}
		}

		with HTTPSingleStoreSaver.from_url(http_base_url, api_key=http_api_key) as saver:
			results = list(saver.list(config))
			assert len(results) == 0

	def test_list_with_before_checkpoint(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test list with before parameter containing checkpoint_id."""
		expected_params = {
			"thread_id": TEST_THREAD_ID,
			"checkpoint_ns": "",
			"before_checkpoint_id": TEST_CHECKPOINT_BEFORE,
		}

		add_mock_response(
			httpx_mock,
			method="GET",
			url=httpx.URL(f"{http_base_url}/checkpoints", params=expected_params),
			json={"checkpoints": []},
		)

		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}
		before = {
			"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": "", "checkpoint_id": TEST_CHECKPOINT_BEFORE}
		}

		with HTTPSingleStoreSaver.from_url(http_base_url, api_key=http_api_key) as saver:
			results = list(saver.list(config, before=before))
			assert len(results) == 0

	def test_list_with_all_parameters(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test list with all optional parameters."""
		import json
		from langgraph.checkpoint.singlestore.http.utils import prepare_metadata_filter

		filter_dict = {"source": "test", "step": 5}
		prepared_filter = prepare_metadata_filter(filter_dict)

		expected_params = {
			"thread_id": TEST_THREAD_ID,
			"checkpoint_ns": "namespace",
			"checkpoint_id": TEST_CHECKPOINT_ID,
			"metadata_filter": json.dumps(prepared_filter),
			"before_checkpoint_id": TEST_CHECKPOINT_BEFORE,
			"limit": 10,
		}

		add_mock_response(
			httpx_mock,
			method="GET",
			url=httpx.URL(f"{http_base_url}/checkpoints", params=expected_params),
			json={"checkpoints": []},
		)

		config = {
			"configurable": {
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "namespace",
				"checkpoint_id": TEST_CHECKPOINT_ID,
			}
		}
		before = {
			"configurable": {
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "namespace",
				"checkpoint_id": TEST_CHECKPOINT_BEFORE,
			}
		}

		with HTTPSingleStoreSaver.from_url(http_base_url, api_key=http_api_key) as saver:
			results = list(saver.list(config, filter=filter_dict, before=before, limit=10))
			assert len(results) == 0

	def test_put_with_primitive_channel_values(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test put with primitive channel values (string, int, float, bool)."""
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
		checkpoint["channel_values"] = {
			"string_ch": "test_string",
			"int_ch": 42,
			"float_ch": 3.14,
			"bool_ch": True,
			"none_ch": None,
		}

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/{checkpoint['id']}",
			json={},
		)

		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}
		with HTTPSingleStoreSaver.from_url(http_base_url, api_key=http_api_key) as saver:
			saver.put(config, checkpoint, {}, {})

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)

			# Primitive values should remain in checkpoint, not in blob_data
			assert "string_ch" in request_body["checkpoint"]["channel_values"]
			assert request_body["checkpoint"]["channel_values"]["string_ch"] == "test_string"
			assert request_body["checkpoint"]["channel_values"]["int_ch"] == 42
			assert request_body["checkpoint"]["channel_values"]["float_ch"] == 3.14
			assert request_body["checkpoint"]["channel_values"]["bool_ch"] is True
			assert request_body["checkpoint"]["channel_values"]["none_ch"] is None

	def test_put_with_mixed_channel_types(
		self,
		http_base_url: str,
		http_api_key: str,
		httpx_mock,
	):
		"""Test put with mixed primitive and complex channel values."""
		checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
		checkpoint["channel_values"] = {
			"simple": "string_value",
			"number": 123,
			"complex": {"nested": "object"},
			"list_value": [1, 2, 3],
			"binary": b"binary_data",
		}

		add_mock_response(
			httpx_mock,
			method="PUT",
			url=f"{http_base_url}/checkpoints/{TEST_THREAD_ID}/{checkpoint['id']}",
			json={},
		)

		config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}
		new_versions = {"complex": "v1", "list_value": "v2", "binary": "v3"}
		with HTTPSingleStoreSaver.from_url(http_base_url, api_key=http_api_key) as saver:
			saver.put(config, checkpoint, {}, new_versions)

		# Verify the request (only in mock mode)
		if httpx_mock:
			request = httpx_mock.get_request()
			request_body = json.loads(request.content)

			# Simple values stay in checkpoint
			assert "simple" in request_body["checkpoint"]["channel_values"]
			assert "number" in request_body["checkpoint"]["channel_values"]

			# Complex values go to blob_data
			assert "blob_data" in request_body
			assert len(request_body["blob_data"]) == 3


@pytest.mark.no_db
class TestPendingWrites:
	"""Test pending writes functionality."""

	def test_multiple_pending_writes(
		self,
		httpx_mock: HTTPXMock,
	):
		"""Test storing multiple pending writes."""
		for _ in range(3):
			httpx_mock.add_response(
				method="PUT",
				url=f"http://localhost:8080/checkpoints/{TEST_THREAD_WRITES}/{TEST_CHECKPOINT_ID}/writes",
				json={},
			)

		config = {
			"configurable": {
				"thread_id": TEST_THREAD_WRITES,
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_ID,
			}
		}

		writes = [
			[("channel1", "value1"), ("channel2", "value2")],
			[("channel3", {"nested": "object"})],
			[("channel4", b"binary data")],
		]

		# Generate UUIDs for task IDs
		task_ids = [generate_uuid_string() for _ in writes]

		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			for task_id, write_batch in zip(task_ids, writes):
				saver.put_writes(config, write_batch, task_id=task_id)

		requests = httpx_mock.get_requests()
		assert len(requests) == 3

		for i, request in enumerate(requests):
			body = json.loads(request.content)
			assert body["task_id"] == task_ids[i]

	def test_writes_with_binary_data(
		self,
		httpx_mock: HTTPXMock,
	):
		"""Test pending writes containing binary data."""
		binary_data = b"\x00\x01\x02\x03\x04"
		encoded = base64.b64encode(binary_data).decode("utf-8")

		httpx_mock.add_response(
			method="PUT",
			url=f"http://localhost:8080/checkpoints/{TEST_THREAD_BINARY_WRITES}/{TEST_CHECKPOINT_ID}/writes",
			json={},
		)

		config = {
			"configurable": {
				"thread_id": TEST_THREAD_BINARY_WRITES,
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_ID,
			}
		}

		writes = [("binary_channel", binary_data)]

		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			saver.put_writes(config, writes, task_id="binary-task")

		request = httpx_mock.get_request()
		body = json.loads(request.content)
		assert len(body["writes"]) == 1
		write = body["writes"][0]
		assert write["channel"] == "binary_channel"
		assert write["blob"] == encoded


@pytest.mark.no_db
class TestValidationErrors:
	"""Test schema validation error handling."""

	def test_list_with_invalid_limit(self, http_base_url: str, http_api_key: str):
		"""Test list with invalid limit value triggers validation error."""
		with HTTPSingleStoreSaver.from_url(http_base_url, api_key=http_api_key) as saver:
			config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

			# Limit must be >= 1 according to schema
			with pytest.raises(HTTPClientError) as exc_info:
				list(saver.list(config, limit=0))

			assert "Invalid request payload" in str(exc_info.value)
			assert exc_info.value.error_code == "INVALID_REQUEST_PAYLOAD"

			# Negative limit should also fail
			with pytest.raises(HTTPClientError) as exc_info:
				list(saver.list(config, limit=-5))

			assert "Invalid request payload" in str(exc_info.value)

	@pytest.mark.mock_only  # Requires mocking malformed response
	def test_list_with_malformed_response(self, httpx_mock: HTTPXMock):
		"""Test list with response that doesn't match schema."""
		httpx_mock.add_response(
			method="GET",
			url=httpx.URL(
				"http://localhost:8080/checkpoints", params={"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}
			),
			json={
				# Missing required "checkpoints" field
				"wrong_field": []
			},
		)

		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

			# Should raise error when trying to parse response
			with pytest.raises(HTTPClientError) as exc_info:
				list(saver.list(config))
			assert "Invalid response from server" in str(exc_info.value)

	@pytest.mark.mock_only  # Requires mocking malformed response
	def test_get_with_malformed_response(self, httpx_mock: HTTPXMock):
		"""Test get with response missing required fields."""
		httpx_mock.add_response(
			method="GET",
			url=f"http://localhost:8080/checkpoints/{TEST_THREAD_ID}/latest",
			json={
				# Missing required fields like thread_id, checkpoint_id, etc.
				"partial_data": "incomplete"
			},
		)

		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

			# Should raise error when trying to parse response
			with pytest.raises(HTTPClientError) as exc_info:
				saver.get_tuple(config)
			assert "Invalid checkpoint response from server" in str(exc_info.value)

	def test_put_with_invalid_checkpoint_data(self, httpx_mock: HTTPXMock):
		"""Test put with checkpoint data that fails validation."""
		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			config = {"configurable": {"thread_id": TEST_THREAD_ID, "checkpoint_ns": ""}}

			# Create checkpoint with invalid data type for version
			invalid_checkpoint = {
				"v": "not_an_int",  # Should be int
				"ts": "2024-01-01T00:00:00Z",
				"id": TEST_CHECKPOINT_ID,
				"channel_values": {},
				"channel_versions": {},
			}

			# This should fail during request model creation
			with pytest.raises((TypeError, ValidationError)):
				saver.put(config, invalid_checkpoint, {}, {})

	@pytest.mark.mock_only  # Requires mocking malformed response
	def test_setup_with_malformed_success_response(self, httpx_mock: HTTPXMock):
		"""Test setup with response that doesn't match expected schema."""
		httpx_mock.add_response(
			method="POST",
			url="http://localhost:8080/checkpoints/setup",
			json={
				# Missing required "success" field
				"result": "ok"
			},
		)

		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			# Should raise error when trying to parse response
			with pytest.raises(HTTPClientError) as exc_info:
				saver.setup()
			assert "Invalid setup response from server" in str(exc_info.value)

	def test_put_writes_with_invalid_task_id_type(self, httpx_mock: HTTPXMock):
		"""Test put_writes with invalid data types."""
		config = {
			"configurable": {
				"thread_id": TEST_THREAD_ID,
				"checkpoint_ns": "",
				"checkpoint_id": TEST_CHECKPOINT_ID,
			}
		}

		with HTTPSingleStoreSaver.from_url("http://localhost:8080", api_key=None) as saver:
			# task_id should be string, not int
			writes = [("channel1", "value1")]

			# This should work - task_id gets converted to string
			httpx_mock.add_response(
				method="PUT",
				url=f"http://localhost:8080/checkpoints/{TEST_THREAD_ID}/{TEST_CHECKPOINT_ID}/writes",
				json={},
			)
			saver.put_writes(config, writes, task_id=123)  # Will be converted to "123"

			request = httpx_mock.get_request()
			body = json.loads(request.content)
			assert body["task_id"] == "123"
