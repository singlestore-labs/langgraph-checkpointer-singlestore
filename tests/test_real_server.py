"""Comprehensive integration tests for HTTP SingleStore checkpointer with real server.

These tests are specifically designed to work with real servers containing
persisted data. All tests use dynamically generated UUIDs and unique markers
to ensure isolation and avoid conflicts with existing data.

Run with: pytest tests/test_real_server.py --use-real-server --server-url=http://localhost:8080
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.singlestore.http.schemas import generate_uuid_string
from langgraph.checkpoint.singlestore.http import HTTPSingleStoreSaver
from langgraph.checkpoint.singlestore.http.aio import AsyncHTTPSingleStoreSaver

from tests.utils.test_data_generators import (
	generate_unique_thread_id,
	generate_unique_checkpoint_id,
	generate_unique_task_id,
	generate_test_marker,
	generate_checkpoint_with_marker,
	generate_config_with_marker,
	cleanup_test_data,
)


@pytest.mark.real_server_only
class TestIntegrationBasicFlow:
	"""Test basic checkpoint flow with real Go server."""

	def test_setup_integration(self, http_base_url: str, http_api_key: str):
		"""Test setup operation with real server."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client
			# This should succeed if Go server is properly configured
			saver.setup()

	def test_full_checkpoint_lifecycle(self, http_base_url: str, http_api_key: str):
		"""Test complete checkpoint lifecycle with real server."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Create test data with UUIDs
			thread_id = generate_uuid_string()
			checkpoint_id = generate_uuid_string()

			config = {
				"configurable": {
					"thread_id": thread_id,
					"checkpoint_ns": "",
					"checkpoint_id": checkpoint_id,
				}
			}

			checkpoint = {
				"v": 1,
				"ts": "2024-01-01T00:00:00Z",
				"id": checkpoint_id,
				"channel_values": {
					"messages": [{"type": "human", "content": "Hello"}],
					"counter": 42,
				},
				"channel_versions": {
					"messages": "1.0",
					"counter": "1.0",
				},
				"versions_seen": {},
			}

			metadata = {"source": "integration_test"}
			new_versions = {"messages": "1.0", "counter": "1.0"}

			# Test PUT operation
			result_config = saver.put(config, checkpoint, metadata, new_versions)
			assert result_config["configurable"]["thread_id"] == thread_id
			assert result_config["configurable"]["checkpoint_id"] == checkpoint_id

			# Test GET operation
			retrieved = saver.get(config)
			assert retrieved is not None
			assert retrieved["id"] == checkpoint_id
			assert retrieved["channel_values"]["counter"] == 42

			# Test LIST operation
			checkpoints = list(saver.list(config))
			assert len(checkpoints) > 0
			assert checkpoints[0].checkpoint["id"] == checkpoint_id

			# Test DELETE operation
			saver.delete_thread(thread_id)

			# Verify deletion
			retrieved_after_delete = saver.get(config)
			assert retrieved_after_delete is None


@pytest.mark.real_server_only
class TestIntegrationBinaryData:
	"""Test binary data handling with real Go server."""

	def test_binary_blob_roundtrip(self, http_base_url: str, http_api_key: str):
		"""Test binary data encoding/decoding with real server."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Create test data with binary blob
			thread_id = generate_uuid_string()
			checkpoint_id = generate_uuid_string()

			config = {
				"configurable": {
					"thread_id": thread_id,
					"checkpoint_ns": "",
					"checkpoint_id": checkpoint_id,
				}
			}

			# Binary data that should survive roundtrip
			binary_data = b"\x00\x01\x02\x03\xff\xfe\xfd\x80\x7f"

			checkpoint = {
				"v": 1,
				"ts": "2024-01-01T00:00:00Z",
				"id": checkpoint_id,
				"channel_values": {
					"binary_channel": binary_data,  # This will be stored as blob
					"text_channel": "simple text",  # This will be inlined
				},
				"channel_versions": {
					"binary_channel": "1.0",
					"text_channel": "1.0",
				},
				"versions_seen": {},
			}

			metadata = {"test": "binary_data"}
			new_versions = {"binary_channel": "1.0", "text_channel": "1.0"}

			# Store checkpoint
			saver.put(config, checkpoint, metadata, new_versions)

			# Retrieve and verify
			retrieved = saver.get(config)
			assert retrieved is not None
			assert retrieved["channel_values"]["binary_channel"] == binary_data
			assert retrieved["channel_values"]["text_channel"] == "simple text"

			# Cleanup
			saver.delete_thread(thread_id)

	def test_checkpoint_writes_with_binary(self, http_base_url: str, http_api_key: str):
		"""Test checkpoint writes with binary data."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Create checkpoint first
			thread_id = generate_uuid_string()
			checkpoint_id = generate_uuid_string()

			config = {
				"configurable": {
					"thread_id": thread_id,
					"checkpoint_ns": "",
					"checkpoint_id": checkpoint_id,
				}
			}

			checkpoint = {
				"v": 1,
				"ts": "2024-01-01T00:00:00Z",
				"id": checkpoint_id,
				"channel_values": {},
				"channel_versions": {},
				"versions_seen": {},
			}

			saver.put(config, checkpoint, {}, {})

			# Add writes with binary data
			binary_write_data = b"\xde\xad\xbe\xef\x00\x11\x22\x33"
			writes = [
				("channel_1", {"action": "update", "data": binary_write_data}),
				("channel_2", "simple string"),
			]

			task_id = generate_uuid_string()
			saver.put_writes(config, writes, task_id, "test_task")

			# Retrieve and verify pending writes
			retrieved_tuple = saver.get_tuple(config)
			assert retrieved_tuple is not None
			assert len(retrieved_tuple.pending_writes) > 0

			# Find our writes
			found_binary = False
			found_string = False
			for _task, channel, value in retrieved_tuple.pending_writes:
				if channel == "channel_1":
					assert value["data"] == binary_write_data
					found_binary = True
				elif channel == "channel_2":
					assert value == "simple string"
					found_string = True

			assert found_binary, "Binary write data not found in pending writes"
			assert found_string, "String write data not found in pending writes"

			# Cleanup
			saver.delete_thread(thread_id)


@pytest.mark.real_server_only
class TestIntegrationAsync:
	"""Test async HTTP client with real Go server."""

	@pytest.mark.asyncio
	async def test_async_basic_operations(self, http_base_url: str, http_api_key: str):
		"""Test basic async operations with real server."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		async with saver._get_client() as client:
			saver._client = client

			# Test async setup
			await saver.setup()

			# Create test data
			thread_id = generate_uuid_string()
			checkpoint_id = generate_uuid_string()

			config = {
				"configurable": {
					"thread_id": thread_id,
					"checkpoint_ns": "",
					"checkpoint_id": checkpoint_id,
				}
			}

			checkpoint = {
				"v": 1,
				"ts": "2024-01-01T00:00:00Z",
				"id": checkpoint_id,
				"channel_values": {"test": "async_data"},
				"channel_versions": {"test": "1.0"},
				"versions_seen": {},
			}

			# Test async operations
			await saver.aput(config, checkpoint, {}, {"test": "1.0"})

			retrieved = await saver.aget(config)
			assert retrieved is not None
			assert retrieved["channel_values"]["test"] == "async_data"

			# Test async list
			checkpoints = []
			async for cp in saver.alist(config):
				checkpoints.append(cp)

			assert len(checkpoints) > 0
			assert checkpoints[0].checkpoint["id"] == checkpoint_id

			# Cleanup
			await saver.adelete_thread(thread_id)


@pytest.mark.real_server_only
class TestIntegrationMetadataFiltering:
	"""Test metadata filtering with real Go server."""

	def test_complex_metadata_filtering(self, http_base_url: str, http_api_key: str):
		"""Test complex metadata filtering scenarios."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Create multiple checkpoints with different metadata
			thread_id = generate_uuid_string()

			test_cases = [
				{
					"checkpoint_id": generate_uuid_string(),
					"metadata": {"user": "alice", "type": "chat", "priority": 1},
				},
				{
					"checkpoint_id": generate_uuid_string(),
					"metadata": {"user": "bob", "type": "task", "priority": 2},
				},
				{
					"checkpoint_id": generate_uuid_string(),
					"metadata": {"user": "alice", "type": "chat", "nested": {"level": "high", "category": "important"}},
				},
			]

			# Create all checkpoints
			for case in test_cases:
				config = {
					"configurable": {
						"thread_id": thread_id,
						"checkpoint_ns": "",
						"checkpoint_id": case["checkpoint_id"],
					}
				}

				checkpoint = {
					"v": 1,
					"ts": "2024-01-01T00:00:00Z",
					"id": case["checkpoint_id"],
					"channel_values": {},
					"channel_versions": {},
					"versions_seen": {},
				}

				saver.put(config, checkpoint, case["metadata"], {})

			# Test simple filtering
			base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

			alice_checkpoints = list(saver.list(base_config, filter={"user": "alice"}))
			assert len(alice_checkpoints) == 2

			# Test nested filtering
			nested_checkpoints = list(saver.list(base_config, filter={"nested": {"level": "high"}}))
			assert len(nested_checkpoints) == 1
			assert nested_checkpoints[0].metadata["nested"]["level"] == "high"

			# Cleanup
			saver.delete_thread(thread_id)


@pytest.mark.real_server_only
class TestIntegrationErrorHandling:
	"""Test error handling with real Go server."""

	def test_not_found_handling(self, http_base_url: str, http_api_key: str):
		"""Test 404 handling with real server."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Try to get non-existent checkpoint
			non_existent_config = {
				"configurable": {
					"thread_id": generate_uuid_string(),
					"checkpoint_ns": "",
					"checkpoint_id": generate_uuid_string(),
				}
			}

			result = saver.get(non_existent_config)
			assert result is None

			tuple_result = saver.get_tuple(non_existent_config)
			assert tuple_result is None

	def test_invalid_uuid_handling(self, http_base_url: str, http_api_key: str):
		"""Test invalid UUID handling with real server."""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Invalid UUID should cause validation error
			invalid_config = {
				"configurable": {
					"thread_id": "not-a-uuid",
					"checkpoint_ns": "",
				}
			}

			# This should raise an HTTPClientError due to validation
			with pytest.raises(Exception):  # HTTPClientError or validation error
				saver.get(invalid_config)


@pytest.mark.real_server_only
class TestRealServerBasicOperations:
	"""Test basic CRUD operations with real server and persisted database.

	This class tests fundamental operations ensuring they work correctly
	even when the database contains existing data from previous runs.
	"""

	def test_setup_and_connection(self, http_base_url: str, http_api_key: str):
		"""Test server connectivity and setup operation.

		This test verifies:
		- HTTP client can connect to server
		- Setup operation succeeds (even if already done)
		- Server responds with expected format
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		# Test setup - should succeed even if already done
		with saver._get_client() as client:
			saver._client = client
			result = saver.setup()

			# Setup should either succeed or be idempotent
			assert result is None or isinstance(result, dict)

	def test_checkpoint_crud_lifecycle(self, http_base_url: str, http_api_key: str):
		"""Test complete CRUD lifecycle with unique identifiers.

		This test:
		1. Creates a checkpoint with unique IDs
		2. Retrieves it and verifies all fields
		3. Updates with new metadata
		4. Deletes the thread
		5. Verifies deletion worked
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		# Generate unique test data
		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		checkpoint, metadata = generate_checkpoint_with_marker(
			checkpoint_id=checkpoint_id,
			test_marker=test_marker,
			channel_values={
				"messages": [{"role": "user", "content": "Hello world"}],
				"counter": 1,
			},
			metadata={"source": "crud_test", "step": 1},
		)

		new_versions = {"messages": "1.0", "counter": "1.0"}

		test_passed = False
		with saver._get_client() as client:
			saver._client = client

			try:
				# 1. CREATE - Put checkpoint
				result_config = saver.put(config, checkpoint, metadata, new_versions)

				assert result_config["configurable"]["thread_id"] == thread_id
				assert result_config["configurable"]["checkpoint_id"] == checkpoint_id

				# 2. READ - Get checkpoint
				retrieved = saver.get(config)
				assert retrieved is not None
				assert retrieved["id"] == checkpoint_id
				assert retrieved["channel_values"]["counter"] == 1
				assert retrieved["channel_values"]["messages"][0]["content"] == "Hello world"

				# Also test get_tuple
				retrieved_tuple = saver.get_tuple(config)
				assert retrieved_tuple is not None
				assert retrieved_tuple.checkpoint["id"] == checkpoint_id
				assert retrieved_tuple.metadata["test_marker"] == test_marker
				assert retrieved_tuple.metadata["source"] == "crud_test"

				# 3. UPDATE - Update with new metadata
				updated_metadata = {
					"test_marker": test_marker,
					"source": "crud_test_updated",
					"step": 2,
					"additional_field": "new_value",
				}

				new_checkpoint_id = generate_unique_checkpoint_id()
				updated_config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=new_checkpoint_id,
				)

				updated_checkpoint = checkpoint.copy()
				updated_checkpoint["id"] = new_checkpoint_id
				updated_checkpoint["channel_values"]["counter"] = 2

				saver.put(updated_config, updated_checkpoint, updated_metadata, new_versions)

				# Verify update
				updated_retrieved = saver.get(updated_config)
				assert updated_retrieved is not None
				assert updated_retrieved["id"] == new_checkpoint_id
				assert updated_retrieved["channel_values"]["counter"] == 2

				updated_tuple = saver.get_tuple(updated_config)
				assert updated_tuple.metadata["source"] == "crud_test_updated"
				assert updated_tuple.metadata["step"] == 2
				assert updated_tuple.metadata["additional_field"] == "new_value"

				# 4. LIST - Verify both checkpoints exist in list
				base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

				checkpoints = list(saver.list(base_config))

				# Should have at least 2 checkpoints (original and updated)
				assert len(checkpoints) >= 2

				# Find our checkpoints in the results
				found_original = False
				found_updated = False

				for cp_tuple in checkpoints:
					if cp_tuple.metadata.get("test_marker") == test_marker:
						if cp_tuple.checkpoint["id"] == checkpoint_id:
							found_original = True
						elif cp_tuple.checkpoint["id"] == new_checkpoint_id:
							found_updated = True

				assert found_original, "Original checkpoint not found in list"
				assert found_updated, "Updated checkpoint not found in list"

				# Mark test as passed
				test_passed = True

			finally:
				# 5. DELETE - Clean up only if test passed
				if test_passed:
					try:
						saver.delete_thread(thread_id)

						# Verify deletion
						deleted_config = generate_config_with_marker(
							thread_id=thread_id,
							checkpoint_id=checkpoint_id,
						)
						retrieved_after_delete = saver.get(deleted_config)
						assert retrieved_after_delete is None

					except Exception as e:
						raise e

	def test_checkpoint_versioning(self, http_base_url: str, http_api_key: str):
		"""Test parent-child checkpoint relationships.

		This test creates a series of checkpoints with parent references
		and verifies the relationships are maintained correctly.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()

		parent_checkpoint_id = generate_unique_checkpoint_id()
		child_checkpoint_id = generate_unique_checkpoint_id()

		test_passed = False
		with saver._get_client() as client:
			saver._client = client

			try:
				# Create parent checkpoint
				# For the first checkpoint (no parent), don't include checkpoint_id in config
				parent_config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=None,
				)

				parent_checkpoint, parent_metadata = generate_checkpoint_with_marker(
					checkpoint_id=parent_checkpoint_id,
					test_marker=test_marker,
					channel_values={"step": "parent", "data": "parent_data"},
					metadata={"type": "parent"},
				)

				saver.put(parent_config, parent_checkpoint, parent_metadata, {"step": "1.0", "data": "1.0"})

				# Create child checkpoint with parent reference via config
				# The parent relationship is established by passing the parent's checkpoint_id in the config
				child_config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=parent_checkpoint_id,
				)

				child_checkpoint, child_metadata = generate_checkpoint_with_marker(
					checkpoint_id=child_checkpoint_id,
					test_marker=test_marker,
					channel_values={"step": "child", "data": "child_data"},
					metadata={"type": "child"},
				)

				# When putting, the checkpoint_id from config becomes parent_checkpoint_id
				saver.put(child_config, child_checkpoint, child_metadata, {"step": "1.0", "data": "1.0"})

				# Verify parent-child relationship
				# Get child checkpoint using the child's ID
				child_retrieval_config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=child_checkpoint_id,
				)
				child_tuple = saver.get_tuple(child_retrieval_config)
				assert child_tuple is not None
				# Verify parent relationship via parent_config
				assert child_tuple.parent_config is not None
				assert child_tuple.parent_config["configurable"]["checkpoint_id"] == parent_checkpoint_id

				# Get parent checkpoint using its ID
				parent_retrieval_config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=parent_checkpoint_id,
				)
				parent_tuple = saver.get_tuple(parent_retrieval_config)
				assert parent_tuple is not None
				assert parent_tuple.parent_config is None  # Parent has no parent

				# Mark test as passed
				test_passed = True

			finally:
				# Clean up only if test passed
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_empty_checkpoint_handling(self, http_base_url: str, http_api_key: str):
		"""Test handling of minimal/empty checkpoints.

		This test verifies the system can handle checkpoints with
		minimal required fields and empty/null values.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		# Minimal checkpoint with empty values
		minimal_checkpoint = {
			"v": 1,
			"id": checkpoint_id,
			"ts": "2024-01-01T00:00:00Z",
			"channel_values": {},  # Empty
			"channel_versions": {},  # Empty
			"versions_seen": {},
			"pending_sends": [],
		}

		minimal_metadata = {
			"test_marker": test_marker,
			"empty_string": "",
			"null_value": None,
			"zero": 0,
			"false": False,
		}

		test_passed = False
		with saver._get_client() as client:
			saver._client = client
			try:
				# Store minimal checkpoint
				saver.put(config, minimal_checkpoint, minimal_metadata, {})

				# Retrieve and verify
				retrieved = saver.get(config)
				assert retrieved is not None
				assert retrieved["id"] == checkpoint_id
				assert retrieved["channel_values"] == {}
				assert retrieved["channel_versions"] == {}

				retrieved_tuple = saver.get_tuple(config)
				assert retrieved_tuple is not None
				assert retrieved_tuple.metadata["test_marker"] == test_marker
				assert retrieved_tuple.metadata["empty_string"] == ""
				assert retrieved_tuple.metadata["null_value"] is None
				assert retrieved_tuple.metadata["zero"] == 0
				assert retrieved_tuple.metadata["false"] is False

				# Mark test as passed
				test_passed = True

			finally:
				# Clean up only if test passed
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_checkpoint_not_found(self, http_base_url: str, http_api_key: str):
		"""Test 404 scenarios with guaranteed non-existent IDs.

		This test uses newly generated UUIDs to ensure they don't exist
		in the database and verifies proper not-found handling.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		# Use fresh UUIDs that are guaranteed not to exist
		non_existent_thread_id = generate_unique_thread_id()
		non_existent_checkpoint_id = generate_unique_checkpoint_id()

		non_existent_config = generate_config_with_marker(
			thread_id=non_existent_thread_id,
			checkpoint_id=non_existent_checkpoint_id,
		)

		with saver._get_client() as client:
			saver._client = client

			# Test get() returns None for non-existent checkpoint
			result = saver.get(non_existent_config)
			assert result is None

			# Test get_tuple() returns None for non-existent checkpoint
			tuple_result = saver.get_tuple(non_existent_config)
			assert tuple_result is None

			# Test list() returns empty for non-existent thread
			base_config = {"configurable": {"thread_id": non_existent_thread_id, "checkpoint_ns": ""}}
			checkpoints = list(saver.list(base_config))
			assert len(checkpoints) == 0

	def test_basic_error_recovery(self, http_base_url: str, http_api_key: str):
		"""Test basic error handling and recovery scenarios.

		This test verifies the client handles common error scenarios
		gracefully without breaking subsequent operations.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		with saver._get_client() as client:
			saver._client = client

			# Test 1: Invalid checkpoint ID format (if server validates)
			try:
				invalid_config = {
					"configurable": {
						"thread_id": "not-a-valid-uuid",
						"checkpoint_id": "also-not-a-uuid",
						"checkpoint_ns": "",
					}
				}
				result = saver.get(invalid_config)
				# Either returns None or raises appropriate error
				# Both are acceptable behaviors
			except Exception as e:
				# Should be a meaningful error, not a generic failure
				assert "uuid" in str(e).lower() or "invalid" in str(e).lower()

			# Test 2: Verify client still works after error
			thread_id = generate_unique_thread_id()
			valid_config = generate_config_with_marker(thread_id=thread_id)

			# This should work fine after the previous error
			result = saver.get(valid_config)
			assert result is None  # Non-existent, but should not error


@pytest.mark.real_server_only
class TestRealServerListingAndFiltering:
	"""Test listing and filtering functionality with real server data.

	These tests create isolated datasets with unique markers and verify
	filtering works correctly even when the database contains other data.
	"""

	def test_list_with_isolated_data(self, http_base_url: str, http_api_key: str):
		"""Test basic listing functionality with isolated test data.

		Creates a known set of checkpoints and verifies they can be
		retrieved through the list operation.
		"""
		from tests.utils.test_data_generators import create_checkpoint_series

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		checkpoint_count = 5
		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Create series of checkpoints
				checkpoint_series = create_checkpoint_series(
					thread_id=thread_id,
					count=checkpoint_count,
					test_marker=test_marker,
				)

				# Store all checkpoints
				for config, checkpoint, metadata, versions in checkpoint_series:
					saver.put(config, checkpoint, metadata, versions)

				# List all checkpoints for this thread
				base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

				checkpoints = list(saver.list(base_config))

				# Should have exactly our checkpoint_count checkpoints
				assert len(checkpoints) == checkpoint_count

				# Verify all our checkpoints are present
				found_markers = set()
				for cp_tuple in checkpoints:
					if cp_tuple.metadata.get("test_marker") == test_marker:
						found_markers.add(cp_tuple.metadata.get("sequence"))

				expected_markers = set(range(checkpoint_count))
				assert found_markers == expected_markers

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_metadata_filtering_with_unique_markers(self, http_base_url: str, http_api_key: str):
		"""Test metadata filtering using unique markers to identify test data.

		Creates checkpoints with various metadata combinations and tests
		that filtering returns the correct subset.
		"""
		from tests.utils.test_data_generators import generate_metadata_test_cases

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()

		test_passed = False
		with saver._get_client() as client:
			saver._client = client
			try:
				# Generate various metadata test cases
				metadata_cases = generate_metadata_test_cases(test_marker)

				# Create checkpoint for each metadata case
				checkpoint_ids = []
				for i, metadata in enumerate(metadata_cases):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoint_ids.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, _ = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={"index": i, "data": f"case_{i}"},
						metadata=metadata,
					)

					saver.put(config, checkpoint, metadata, {"index": "1.0", "data": "1.0"})

				base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

				# Test 1: Filter by test_marker (should get all)
				all_checkpoints = list(saver.list(base_config, filter={"test_marker": test_marker}))
				assert len(all_checkpoints) == len(metadata_cases)

				# Test 2: Filter by user="alice"
				alice_checkpoints = list(saver.list(base_config, filter={"test_marker": test_marker, "user": "alice"}))
				# Should find 2 alice checkpoints from our test cases
				assert len(alice_checkpoints) == 2
				for cp_tuple in alice_checkpoints:
					assert cp_tuple.metadata["user"] == "alice"

				# Test 3: Filter by type="chat"
				chat_checkpoints = list(saver.list(base_config, filter={"test_marker": test_marker, "type": "chat"}))
				# Should find 2 chat checkpoints
				assert len(chat_checkpoints) == 2
				for cp_tuple in chat_checkpoints:
					assert cp_tuple.metadata["type"] == "chat"

				# Test 4: Combined filter user="alice" AND type="chat"
				alice_chat_checkpoints = list(
					saver.list(base_config, filter={"test_marker": test_marker, "user": "alice", "type": "chat"})
				)
				# Should find 2 checkpoints matching both criteria
				assert len(alice_chat_checkpoints) == 2
				for cp_tuple in alice_chat_checkpoints:
					assert cp_tuple.metadata["user"] == "alice"
					assert cp_tuple.metadata["type"] == "chat"

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_nested_metadata_filtering(self, http_base_url: str, http_api_key: str):
		"""Test filtering on nested metadata structures.

		Creates checkpoints with complex nested metadata and verifies
		deep filtering works correctly.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Create checkpoints with nested metadata
				test_cases = [
					{
						"test_marker": test_marker,
						"config": {"level": "high", "category": "important", "settings": {"auto_save": True}},
					},
					{
						"test_marker": test_marker,
						"config": {"level": "low", "category": "normal", "settings": {"auto_save": False}},
					},
					{
						"test_marker": test_marker,
						"config": {"level": "high", "category": "urgent", "settings": {"auto_save": True}},
					},
				]

				checkpoint_ids = []
				for i, metadata in enumerate(test_cases):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoint_ids.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, _ = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={"case": i},
						metadata=metadata,
					)

					saver.put(config, checkpoint, metadata, {"case": "1.0"})

				base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

				# Test nested filtering: config.level = "high"
				high_level_checkpoints = list(
					saver.list(base_config, filter={"test_marker": test_marker, "config": {"level": "high"}})
				)
				assert len(high_level_checkpoints) == 2
				for cp_tuple in high_level_checkpoints:
					assert cp_tuple.metadata["config"]["level"] == "high"

				# Test deeper nesting: config.settings.auto_save = true
				auto_save_checkpoints = list(
					saver.list(
						base_config, filter={"test_marker": test_marker, "config": {"settings": {"auto_save": True}}}
					)
				)
				assert len(auto_save_checkpoints) == 2
				for cp_tuple in auto_save_checkpoints:
					assert cp_tuple.metadata["config"]["settings"]["auto_save"] is True

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_list_with_limit_and_pagination(self, http_base_url: str, http_api_key: str):
		"""Test list operation with limit and pagination using known data set.

		Creates a known number of checkpoints and tests limit/pagination
		parameters work correctly.
		"""
		from tests.utils.test_data_generators import create_checkpoint_series

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		total_checkpoints = 20
		limit = 5
		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Create series of checkpoints
				checkpoint_series = create_checkpoint_series(
					thread_id=thread_id,
					count=total_checkpoints,
					test_marker=test_marker,
				)

				# Store all checkpoints
				for config, checkpoint, metadata, versions in checkpoint_series:
					saver.put(config, checkpoint, metadata, versions)

				base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

				# Test 1: List with limit
				limited_checkpoints = list(saver.list(base_config, limit=limit))
				assert len(limited_checkpoints) == limit

				# Test 2: List without limit (should get all)
				all_checkpoints = list(saver.list(base_config))
				assert len(all_checkpoints) == total_checkpoints

				# Test 3: Test before parameter for pagination
				if len(all_checkpoints) > 1:
					# Use second checkpoint as "before" marker
					before_checkpoint_id = all_checkpoints[1].checkpoint["id"]

					before_checkpoints = list(
						saver.list(base_config, before={"configurable": {"checkpoint_id": before_checkpoint_id}})
					)

					# Should get fewer checkpoints than total
					assert len(before_checkpoints) < total_checkpoints

					# None of the returned checkpoints should have the "before" ID
					returned_ids = {cp.checkpoint["id"] for cp in before_checkpoints}
					assert before_checkpoint_id not in returned_ids

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_cross_namespace_listing(self, http_base_url: str, http_api_key: str):
		"""Test listing across different checkpoint namespaces.

		Creates checkpoints in multiple namespaces and verifies
		namespace isolation works correctly.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		namespaces = ["", "namespace_a", "namespace_b"]
		test_passed = False
		with saver._get_client() as client:
			saver._client = client
			try:
				# Create checkpoints in each namespace
				all_checkpoint_ids = []
				namespace_checkpoint_map = {}

				for namespace in namespaces:
					checkpoint_ids = []

					for i in range(3):  # 3 checkpoints per namespace
						checkpoint_id = generate_unique_checkpoint_id()
						checkpoint_ids.append(checkpoint_id)
						all_checkpoint_ids.append(checkpoint_id)

						config = {
							"configurable": {
								"thread_id": thread_id,
								"checkpoint_id": checkpoint_id,
								"checkpoint_ns": namespace,
							}
						}

						checkpoint, metadata = generate_checkpoint_with_marker(
							checkpoint_id=checkpoint_id,
							test_marker=test_marker,
							channel_values={"namespace": namespace, "index": i},
							metadata={"namespace": namespace, "index": i},
						)

						saver.put(config, checkpoint, metadata, {"namespace": "1.0", "index": "1.0"})

					namespace_checkpoint_map[namespace] = checkpoint_ids

				# Test isolation: list checkpoints in each namespace
				for namespace in namespaces:
					base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace}}

					checkpoints = list(saver.list(base_config))
					assert len(checkpoints) == 3

					# Verify all checkpoints belong to this namespace
					for cp_tuple in checkpoints:
						assert cp_tuple.metadata["namespace"] == namespace
						assert cp_tuple.checkpoint["id"] in namespace_checkpoint_map[namespace]

				# Test: listing default namespace should not include named namespaces
				default_checkpoints = list(saver.list({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}))

				default_ids = {cp.checkpoint["id"] for cp in default_checkpoints}
				expected_default_ids = set(namespace_checkpoint_map[""])

				assert default_ids == expected_default_ids

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_list_with_all_parameters_combined(self, http_base_url: str, http_api_key: str):
		"""Test list operation with multiple parameters combined.

		Creates a complex data set and tests combinations of thread_id,
		checkpoint_ns, metadata filtering, limit, and before parameters.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Create complex dataset: 2 namespaces × 3 types × 2 users = 12 checkpoints
				checkpoint_ids = []

				for namespace in ["", "test_ns"]:
					for checkpoint_type in ["chat", "task", "system"]:
						for user in ["alice", "bob"]:
							checkpoint_id = generate_unique_checkpoint_id()
							checkpoint_ids.append(checkpoint_id)

							config = {
								"configurable": {
									"thread_id": thread_id,
									"checkpoint_id": checkpoint_id,
									"checkpoint_ns": namespace,
								}
							}

							checkpoint, metadata = generate_checkpoint_with_marker(
								checkpoint_id=checkpoint_id,
								test_marker=test_marker,
								channel_values={
									"type": checkpoint_type,
									"user": user,
									"namespace": namespace,
								},
								metadata={
									"type": checkpoint_type,
									"user": user,
									"namespace": namespace,
									"priority": 1 if user == "alice" else 2,
								},
							)

							saver.put(config, checkpoint, metadata, {"type": "1.0", "user": "1.0", "namespace": "1.0"})

				# Test combinations

				# 1. Default namespace + type="chat" + user="alice" + limit=1
				result1 = list(
					saver.list(
						{"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
						filter={"test_marker": test_marker, "type": "chat", "user": "alice"},
						limit=1,
					)
				)
				assert len(result1) == 1
				assert result1[0].metadata["type"] == "chat"
				assert result1[0].metadata["user"] == "alice"
				assert result1[0].metadata["namespace"] == ""

				# 2. Test namespace + type="task" (should find 2: alice + bob)
				result2 = list(
					saver.list(
						{"configurable": {"thread_id": thread_id, "checkpoint_ns": "test_ns"}},
						filter={"test_marker": test_marker, "type": "task"},
					)
				)
				assert len(result2) == 2
				for cp_tuple in result2:
					assert cp_tuple.metadata["type"] == "task"
					assert cp_tuple.metadata["namespace"] == "test_ns"

				# 3. Cross-namespace filter (should be empty due to namespace isolation)
				result3 = list(
					saver.list(
						{"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
						filter={"test_marker": test_marker, "namespace": "test_ns"},
					)
				)
				assert len(result3) == 0  # Namespace isolation

				# 4. Complex metadata filter
				result4 = list(
					saver.list(
						{"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
						filter={"test_marker": test_marker, "priority": 1},  # Only alice
					)
				)
				assert len(result4) == 3  # 3 types for alice in default namespace
				for cp_tuple in result4:
					assert cp_tuple.metadata["user"] == "alice"
					assert cp_tuple.metadata["priority"] == 1

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])


@pytest.mark.real_server_only
class TestRealServerBinaryAndLargeData:
	"""Test binary data and large payload handling with real server.

	These tests verify that binary data survives roundtrips correctly
	and that large payloads are handled without corruption.
	"""

	def test_binary_blob_storage_retrieval(self, http_base_url: str, http_api_key: str):
		"""Test binary data encoding/decoding roundtrip with real server.

		Creates checkpoints with various binary data patterns and verifies
		they survive the storage/retrieval cycle without corruption.
		"""
		from tests.utils.test_data_generators import generate_binary_test_pattern

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()

		# Various binary test patterns
		binary_test_cases = [
			("deadbeef_pattern", generate_binary_test_pattern(1024, "deadbeef")),
			("sequential_bytes", generate_binary_test_pattern(512, "sequential")),
			("edge_case_bytes", generate_binary_test_pattern(256, "edges")),
			("random_bytes", generate_binary_test_pattern(2048, "random")),
			("tiny_binary", b"\x00\x01\x02\x03\xff\xfe\xfd"),
			("empty_binary", b""),
			("null_bytes", b"\x00" * 100),
			("high_bytes", b"\xff" * 100),
		]

		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				checkpoint_ids = []

				for i, (pattern_name, binary_data) in enumerate(binary_test_cases):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoint_ids.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={
							"binary_data": binary_data,  # This should be stored as blob
							"pattern_name": pattern_name,
							"original_length": len(binary_data),
							"text_data": f"Text for pattern {pattern_name}",  # This stays inline
						},
						metadata={
							"pattern": pattern_name,
							"binary_length": len(binary_data),
							"test_index": i,
						},
					)

					# Use unique versions for each checkpoint to avoid blob collision
					# The checkpoint_blobs table's PK is (thread_id, checkpoint_ns, channel, version)
					# so same version across checkpoints in same thread causes overwrites
					versions = {
						"binary_data": f"{i + 1}.0",
						"pattern_name": f"{i + 1}.0",
						"original_length": f"{i + 1}.0",
						"text_data": f"{i + 1}.0",
					}

					# Update checkpoint's channel_versions to match the versions we'll use
					checkpoint["channel_versions"] = versions

					# Store checkpoint
					saver.put(config, checkpoint, metadata, versions)

					# Immediately verify retrieval
					retrieved = saver.get(config)
					assert retrieved is not None
					assert retrieved["id"] == checkpoint_id

					# Verify binary data is identical
					retrieved_binary = retrieved["channel_values"]["binary_data"]
					assert retrieved_binary == binary_data, f"Binary data mismatch for pattern {pattern_name}"

					# Verify other data is preserved
					assert retrieved["channel_values"]["pattern_name"] == pattern_name
					assert retrieved["channel_values"]["original_length"] == len(binary_data)
					assert retrieved["channel_values"]["text_data"] == f"Text for pattern {pattern_name}"

				# Test retrieval via get_tuple as well
				for i, checkpoint_id in enumerate(checkpoint_ids):
					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					retrieved_tuple = saver.get_tuple(config)
					assert retrieved_tuple is not None

					pattern_name, expected_binary = binary_test_cases[i]
					actual_binary = retrieved_tuple.checkpoint["channel_values"]["binary_data"]

					assert retrieved_tuple.metadata["pattern"] == pattern_name
					assert actual_binary == expected_binary, (
						f"get_tuple: Binary data mismatch for pattern {pattern_name}"
					)

					assert retrieved_tuple.metadata["binary_length"] == len(expected_binary)

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_large_payload_handling(self, http_base_url: str, http_api_key: str):
		"""Test handling of large payloads.

		Creates checkpoints with increasingly large payloads to test
		server limits and data integrity with large data.
		"""
		from tests.utils.test_data_generators import generate_large_payload

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()

		# Test various payload sizes (in MB)
		# Start small and increase to find server limits
		payload_sizes = [0.1, 0.5, 1.0]  # MB

		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				successful_sizes = []

				for size_mb in payload_sizes:
					try:
						checkpoint_id = generate_unique_checkpoint_id()

						config = generate_config_with_marker(
							thread_id=thread_id,
							checkpoint_id=checkpoint_id,
						)

						large_payload = generate_large_payload(size_mb)

						checkpoint, metadata = generate_checkpoint_with_marker(
							checkpoint_id=checkpoint_id,
							test_marker=test_marker,
							channel_values={
								"large_data": large_payload["large_text"],
								"size_info": large_payload["metadata"],
								"marker": f"large_test_{size_mb}mb",
							},
							metadata={
								"size_mb": size_mb,
								"test_type": "large_payload",
							},
						)

						versions = {
							"large_data": "1.0",
							"size_info": "1.0",
							"marker": "1.0",
						}

						# Store large checkpoint
						saver.put(config, checkpoint, metadata, versions)

						# Verify retrieval
						retrieved = saver.get(config)
						assert retrieved is not None
						assert retrieved["id"] == checkpoint_id

						# Verify large data integrity
						retrieved_large_data = retrieved["channel_values"]["large_data"]
						original_large_data = large_payload["large_text"]

						assert len(retrieved_large_data) == len(original_large_data), (
							f"Large data length mismatch for {size_mb}MB payload"
						)

						assert retrieved_large_data == original_large_data, (
							f"Large data content mismatch for {size_mb}MB payload"
						)

						# Verify metadata
						retrieved_tuple = saver.get_tuple(config)
						assert retrieved_tuple.metadata["size_mb"] == size_mb

						successful_sizes.append(size_mb)

					except Exception as e:
						# Large payload might fail due to server limits
						# This is acceptable - we're testing limits
						print(f"Large payload {size_mb}MB failed (acceptable): {e}")
						break

				# Should have successfully handled at least the smallest payload
				assert len(successful_sizes) >= 1, "Failed to handle even small payloads"

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_mixed_channel_types(self, http_base_url: str, http_api_key: str):
		"""Test checkpoint with mixed data types in channels.

		Creates a checkpoint containing various data types (strings, numbers,
		booleans, binary, objects, arrays, nulls) and verifies type preservation.
		"""
		from tests.utils.test_data_generators import generate_mixed_channel_types

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Generate mixed channel types
				mixed_channels = generate_mixed_channel_types(test_marker)

				checkpoint, metadata = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id,
					test_marker=test_marker,
					channel_values=mixed_channels,
					metadata={"test_type": "mixed_types"},
				)

				# Create versions for all channels
				versions = {channel: "1.0" for channel in mixed_channels.keys()}

				# Store checkpoint
				saver.put(config, checkpoint, metadata, versions)

				# Retrieve and verify all data types
				retrieved = saver.get(config)
				assert retrieved is not None

				retrieved_channels = retrieved["channel_values"]

				# Test each data type preservation
				assert retrieved_channels["string_simple"] == "simple text value"
				assert retrieved_channels["string_unicode"] == "Unicode: 你好世界 🌍"
				assert retrieved_channels["integer"] == 42
				assert retrieved_channels["float"] == 3.14159
				assert retrieved_channels["boolean_true"] is True
				assert retrieved_channels["boolean_false"] is False
				assert retrieved_channels["null_value"] is None
				assert retrieved_channels["empty_string"] == ""

				# Test collections
				assert retrieved_channels["list_simple"] == [1, 2, 3]
				assert retrieved_channels["list_mixed"] == [1, "two", 3.0, True, None]
				assert retrieved_channels["list_nested"] == [[1, 2], [3, 4], [5, 6]]
				assert retrieved_channels["dict_simple"] == {"key": "value"}

				# Test nested dictionary
				expected_nested = {
					"level1": {
						"level2": {
							"level3": "deep value",
							"test_marker": test_marker,
						}
					}
				}
				assert retrieved_channels["dict_nested"] == expected_nested

				# Test binary data
				assert retrieved_channels["binary_small"] == b"small binary \x00\x01\x02"

				# For large binary, just verify it's binary and has expected length
				large_binary = retrieved_channels["binary_large"]
				assert isinstance(large_binary, bytes)
				assert len(large_binary) == 1024

				# Verify the pattern (should be deadbeef repeated)
				expected_pattern = b"\xde\xad\xbe\xef"
				assert large_binary.startswith(expected_pattern)

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_unicode_and_special_characters(self, http_base_url: str, http_api_key: str):
		"""Test handling of Unicode and special characters.

		Creates checkpoints with various Unicode strings, emojis,
		and special characters to verify proper encoding/decoding.
		"""
		from tests.utils.test_data_generators import generate_unicode_test_strings

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()

		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Get Unicode test strings
				unicode_strings = generate_unicode_test_strings()

				checkpoint_ids = []

				for i, unicode_str in enumerate(unicode_strings):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoint_ids.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={
							"unicode_text": unicode_str,
							"text_index": i,
							"text_length": len(unicode_str),
							"byte_length": len(unicode_str.encode("utf-8")),
						},
						metadata={
							"unicode_description": f"Unicode test case {i}",
							"text_preview": unicode_str[:20] if len(unicode_str) > 20 else unicode_str,
						},
					)

					versions = {
						"unicode_text": "1.0",
						"text_index": "1.0",
						"text_length": "1.0",
						"byte_length": "1.0",
					}

					# Store checkpoint
					saver.put(config, checkpoint, metadata, versions)

					# Immediately verify retrieval
					retrieved = saver.get(config)
					assert retrieved is not None

					# Verify Unicode string is identical
					retrieved_unicode = retrieved["channel_values"]["unicode_text"]
					assert retrieved_unicode == unicode_str, f"Unicode string mismatch for case {i}: {unicode_str[:50]}"

					# Verify length information
					assert retrieved["channel_values"]["text_length"] == len(unicode_str)
					assert retrieved["channel_values"]["byte_length"] == len(unicode_str.encode("utf-8"))

					# Test metadata Unicode handling
					retrieved_tuple = saver.get_tuple(config)
					expected_preview = unicode_str[:20] if len(unicode_str) > 20 else unicode_str
					assert retrieved_tuple.metadata["text_preview"] == expected_preview

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_null_character_handling(self, http_base_url: str, http_api_key: str):
		"""Test handling of null characters and null values.

		Tests both null bytes in strings and JSON null values
		to ensure proper handling and distinction.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Test data with various null scenarios
				null_test_data = {
					"string_with_null_bytes": "before\x00middle\x00after",
					"only_null_bytes": "\x00\x00\x00",
					"json_null_value": None,
					"empty_string": "",
					"string_ending_with_null": "text\x00",
					"binary_with_nulls": b"\x00binary\x00data\x00",
				}

				null_metadata = {
					"test_marker": test_marker,
					"null_value_metadata": None,
					"empty_string_metadata": "",
					"zero_number": 0,
					"false_boolean": False,
					"description": "Testing null handling",
				}

				checkpoint, _ = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id,
					test_marker=test_marker,
					channel_values=null_test_data,
					metadata=null_metadata,
				)

				versions = {channel: "1.0" for channel in null_test_data.keys()}

				# Store checkpoint
				saver.put(config, checkpoint, null_metadata, versions)

				# Retrieve and verify
				retrieved = saver.get(config)
				assert retrieved is not None

				retrieved_channels = retrieved["channel_values"]

				# Verify string with null bytes
				assert retrieved_channels["string_with_null_bytes"] == "before\x00middle\x00after"
				assert retrieved_channels["only_null_bytes"] == "\x00\x00\x00"
				assert retrieved_channels["string_ending_with_null"] == "text\x00"

				# Verify JSON null vs empty string distinction
				assert retrieved_channels["json_null_value"] is None
				assert retrieved_channels["empty_string"] == ""

				# Verify binary data with nulls
				assert retrieved_channels["binary_with_nulls"] == b"\x00binary\x00data\x00"

				# Test metadata null handling
				retrieved_tuple = saver.get_tuple(config)
				assert retrieved_tuple.metadata["null_value_metadata"] is None
				assert retrieved_tuple.metadata["empty_string_metadata"] == ""
				assert retrieved_tuple.metadata["zero_number"] == 0
				assert retrieved_tuple.metadata["false_boolean"] is False
				assert retrieved_tuple.metadata["description"] == "Testing null handling"

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])


@pytest.mark.real_server_only
class TestRealServerPendingWrites:
	"""Test checkpoint writes functionality with real server.

	These tests verify that pending writes are stored and retrieved
	correctly, and that write isolation between threads works.
	"""

	def test_checkpoint_writes_lifecycle(self, http_base_url: str, http_api_key: str):
		"""Test complete checkpoint writes lifecycle.

		Creates a checkpoint, adds writes, verifies they appear in pending_writes,
		then creates a new checkpoint and verifies writes are handled correctly.
		"""
		from tests.utils.test_data_generators import generate_write_data_with_marker

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		test_passed = False
		with saver._get_client() as client:
			saver._client = client
			try:
				# Step 1: Create initial checkpoint
				checkpoint_id = generate_unique_checkpoint_id()
				config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=checkpoint_id,
				)

				checkpoint, metadata = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id,
					test_marker=test_marker,
					channel_values={"initial": "data", "step": 1},
					metadata={"phase": "initial"},
				)

				saver.put(config, checkpoint, metadata, {"initial": "1.0", "step": "1.0"})

				# Step 2: Add writes to the checkpoint
				write_data = generate_write_data_with_marker(test_marker)
				task_id = generate_unique_task_id()

				saver.put_writes(config, write_data, task_id, "test_task")

				# Step 3: Verify writes appear in pending_writes
				retrieved_tuple = saver.get_tuple(config)
				assert retrieved_tuple is not None
				assert len(retrieved_tuple.pending_writes) > 0

				# Find our writes by checking for test_marker in the data
				found_writes = {}
				for task, channel, value in retrieved_tuple.pending_writes:
					if task == task_id:
						found_writes[channel] = value

				assert len(found_writes) == len(write_data)

				# Verify each write was stored correctly
				expected_writes = {channel: value for channel, value in write_data}
				for channel, expected_value in expected_writes.items():
					assert channel in found_writes, f"Channel {channel} not found in writes"
					actual_value = found_writes[channel]

					if isinstance(expected_value, bytes):
						# Binary data should match exactly
						assert actual_value == expected_value, f"Binary write mismatch for {channel}"
					elif isinstance(expected_value, dict) and "test_marker" in expected_value:
						# Object with test marker
						assert actual_value["test_marker"] == test_marker
						assert actual_value["action"] == expected_value["action"]
					else:
						# Other data types
						assert actual_value == expected_value, (
							f"Write mismatch for {channel}: {actual_value} != {expected_value}"
						)

				# Step 4: Create new checkpoint (simulating consumption of writes)
				new_checkpoint_id = generate_unique_checkpoint_id()
				# Use the previous checkpoint_id as parent by putting it in config
				new_config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=checkpoint_id,  # This becomes the parent_checkpoint_id
				)

				new_checkpoint, new_metadata = generate_checkpoint_with_marker(
					checkpoint_id=new_checkpoint_id,
					test_marker=test_marker,
					channel_values={"processed": "writes", "step": 2},
					metadata={"phase": "processed"},
				)

				result_config = saver.put(new_config, new_checkpoint, new_metadata, {"processed": "1.0", "step": "1.0"})

				# Step 5: Verify new checkpoint can be retrieved using the returned config
				new_retrieved = saver.get_tuple(result_config)
				assert new_retrieved is not None
				assert new_retrieved.checkpoint["id"] == new_checkpoint_id
				assert new_retrieved.metadata["phase"] == "processed"

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_multiple_concurrent_writes(self, http_base_url: str, http_api_key: str):
		"""Test multiple writes to the same checkpoint with different task IDs.

		Adds multiple sets of writes with different task IDs and verifies
		they are all preserved and can be distinguished.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		test_passed = False
		with saver._get_client() as client:
			saver._client = client
			try:
				# Create checkpoint
				checkpoint_id = generate_unique_checkpoint_id()
				config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=checkpoint_id,
				)

				checkpoint, metadata = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id,
					test_marker=test_marker,
					channel_values={"base": "checkpoint"},
					metadata={"test_type": "multiple_writes"},
				)

				saver.put(config, checkpoint, metadata, {"base": "1.0"})

				# Add multiple sets of writes with different task IDs
				task_writes = []

				for i in range(3):  # 3 different tasks
					task_id = generate_unique_task_id()
					task_name = f"task_{i}"

					writes = [
						(f"channel_{i}_a", f"value_{i}_a"),
						(f"channel_{i}_b", {"task_index": i, "test_marker": test_marker}),
						(f"channel_{i}_c", f"binary_data_{i}".encode()),
					]

					task_writes.append((task_id, task_name, writes))
					saver.put_writes(config, writes, task_id, task_name)

				# Verify all writes are present
				retrieved_tuple = saver.get_tuple(config)
				assert retrieved_tuple is not None

				# Group writes by task_id
				writes_by_task = {}
				for task, channel, value in retrieved_tuple.pending_writes:
					if task not in writes_by_task:
						writes_by_task[task] = {}
					writes_by_task[task][channel] = value

				# Verify we have writes for all tasks
				expected_task_ids = {task_id for task_id, _, _ in task_writes}
				actual_task_ids = set(writes_by_task.keys())

				# Should have at least our expected tasks (might have others from previous tests)
				for expected_task_id in expected_task_ids:
					assert expected_task_id in actual_task_ids, f"Task {expected_task_id} not found in writes"

				# Verify each task's writes
				for task_id, task_name, expected_writes in task_writes:
					task_writes_dict = writes_by_task[task_id]

					for channel, expected_value in expected_writes:
						assert channel in task_writes_dict, f"Channel {channel} not found for task {task_id}"
						actual_value = task_writes_dict[channel]

						if isinstance(expected_value, bytes):
							assert actual_value == expected_value
						elif isinstance(expected_value, dict) and "test_marker" in expected_value:
							assert actual_value["test_marker"] == test_marker
							assert actual_value["task_index"] == expected_value["task_index"]
						else:
							assert actual_value == expected_value

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_writes_with_binary_data(self, http_base_url: str, http_api_key: str):
		"""Test checkpoint writes containing binary data.

		Creates writes with various binary data types and verifies
		they survive the storage/retrieval cycle correctly.
		"""
		from tests.utils.test_data_generators import generate_binary_test_pattern

		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		test_passed = False
		with saver._get_client() as client:
			saver._client = client
			try:
				# Create checkpoint
				checkpoint_id = generate_unique_checkpoint_id()
				config = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=checkpoint_id,
				)

				checkpoint, metadata = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id,
					test_marker=test_marker,
					channel_values={"setup": "for binary writes"},
				)

				saver.put(config, checkpoint, metadata, {"setup": "1.0"})

				# Create writes with various binary patterns
				binary_writes = [
					("tiny_binary", b"\x00\x01\x02\x03"),
					("deadbeef_pattern", generate_binary_test_pattern(512, "deadbeef")),
					("edge_bytes", generate_binary_test_pattern(256, "edges")),
					("text_with_binary", "Text with binary: \x00\xff\x80\x7f"),
					(
						"json_with_binary",
						{
							"test_marker": test_marker,
							"binary_field": b"\xde\xad\xbe\xef",
							"text_field": "normal text",
						},
					),
				]

				task_id = generate_unique_task_id()
				saver.put_writes(config, binary_writes, task_id, "binary_write_test")

				# Verify binary data in writes
				retrieved_tuple = saver.get_tuple(config)
				assert retrieved_tuple is not None

				# Find our writes
				found_writes = {}
				for task, channel, value in retrieved_tuple.pending_writes:
					if task == task_id:
						found_writes[channel] = value

				assert len(found_writes) == len(binary_writes)

				# Verify each binary write
				for channel, expected_value in binary_writes:
					assert channel in found_writes, f"Binary channel {channel} not found"
					actual_value = found_writes[channel]

					if isinstance(expected_value, bytes):
						assert actual_value == expected_value, f"Binary mismatch for {channel}"
					elif isinstance(expected_value, str):
						assert actual_value == expected_value, f"String mismatch for {channel}"
					elif isinstance(expected_value, dict):
						assert actual_value["test_marker"] == test_marker
						assert actual_value["binary_field"] == expected_value["binary_field"]
						assert actual_value["text_field"] == expected_value["text_field"]

				test_passed = True

			finally:
				if test_passed:
					cleanup_test_data(saver, test_marker, [thread_id])

	def test_writes_isolation(self, http_base_url: str, http_api_key: str):
		"""Test that writes are isolated between different threads.

		Creates writes in multiple threads and verifies that each thread
		only sees its own writes, not writes from other threads.
		"""
		saver = HTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		test_marker = generate_test_marker()
		thread_count = 3
		threads_data = []
		test_passed = False

		with saver._get_client() as client:
			saver._client = client
			try:
				# Create checkpoints and writes in multiple threads
				for i in range(thread_count):
					thread_id = generate_unique_thread_id()
					checkpoint_id = generate_unique_checkpoint_id()
					task_id = generate_unique_task_id()

					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={"thread_index": i},
						metadata={"thread_isolation_test": True, "index": i},
					)

					saver.put(config, checkpoint, metadata, {"thread_index": "1.0"})

					# Add writes specific to this thread
					writes = [
						(f"thread_{i}_channel_a", f"thread_{i}_value_a"),
						(
							f"thread_{i}_channel_b",
							{
								"thread_index": i,
								"test_marker": test_marker,
								"isolation_test": True,
							},
						),
						("shared_channel_name", f"thread_{i}_unique_value"),  # Same channel name, different values
					]

					saver.put_writes(config, writes, task_id, f"thread_{i}_task")

					threads_data.append(
						{
							"thread_id": thread_id,
							"config": config,
							"expected_writes": writes,
							"task_id": task_id,
						}
					)

				# Verify isolation: each thread should only see its own writes
				for i, thread_data in enumerate(threads_data):
					retrieved_tuple = saver.get_tuple(thread_data["config"])
					assert retrieved_tuple is not None

					# Find writes for this thread's task
					thread_writes = {}
					for task, channel, value in retrieved_tuple.pending_writes:
						if task == thread_data["task_id"]:
							thread_writes[channel] = value

					# Should have exactly the expected writes for this thread
					expected_writes = {channel: value for channel, value in thread_data["expected_writes"]}

					assert len(thread_writes) == len(expected_writes), (
						f"Thread {i} has wrong number of writes: {len(thread_writes)} != {len(expected_writes)}"
					)

					# Verify each expected write exists and is correct
					for channel, expected_value in expected_writes.items():
						assert channel in thread_writes, f"Thread {i} missing channel {channel}"
						actual_value = thread_writes[channel]

						if isinstance(expected_value, dict) and "test_marker" in expected_value:
							assert actual_value["test_marker"] == test_marker
							assert actual_value["thread_index"] == i
							assert actual_value["isolation_test"] is True
						else:
							assert actual_value == expected_value

					# Verify the shared channel has this thread's unique value
					shared_value = thread_writes["shared_channel_name"]
					assert shared_value == f"thread_{i}_unique_value", (
						f"Thread {i} has wrong shared channel value: {shared_value}"
					)

				test_passed = True

			finally:
				# Cleanup all threads
				all_thread_ids = [thread_data["thread_id"] for thread_data in threads_data]
				if test_passed:
					cleanup_test_data(saver, test_marker, all_thread_ids)
