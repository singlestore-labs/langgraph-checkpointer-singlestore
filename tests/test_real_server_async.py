"""Async integration tests for AsyncHTTPSingleStoreSaver with real server."""

import asyncio
import hashlib
import time
from typing import Any, Dict
from unittest.mock import patch

import pytest
from httpx import TimeoutException

from langgraph.checkpoint.singlestore.http.aio import AsyncHTTPSingleStoreSaver
from langgraph.checkpoint.singlestore.http.schemas import generate_uuid_string

from tests.utils.test_data_generators import (
	generate_binary_test_pattern,
	generate_checkpoint_with_marker,
	generate_config_with_marker,
	generate_test_marker,
	generate_unique_checkpoint_id,
	generate_unique_thread_id,
)


@pytest.mark.asyncio
@pytest.mark.real_server_only
class TestAsyncIntegrationBasicFlow:
	"""Test async basic flow with real Go server."""

	async def test_async_setup_operations(self, http_base_url: str, http_api_key: str):
		"""Test async setup operations."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		# Test setup - should succeed even if already done
		async with saver:
			result = await saver.setup()

			# Setup should either succeed or be idempotent
			assert result is None or isinstance(result, dict)

	async def test_async_checkpoint_crud_lifecycle(self, http_base_url: str, http_api_key: str):
		"""Test complete async CRUD lifecycle with unique identifiers."""
		saver = AsyncHTTPSingleStoreSaver(
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
				"messages": [{"role": "user", "content": "Hello async world"}],
				"counter": 1,
			},
			metadata={"source": "async_crud_test", "step": 1},
		)

		new_versions = {"messages": "1.0", "counter": "1.0"}

		test_passed = False
		async with saver:
			try:
				# 1. CREATE - Put checkpoint
				result_config = await saver.aput(config, checkpoint, metadata, new_versions)

				assert result_config["configurable"]["thread_id"] == thread_id
				assert result_config["configurable"]["checkpoint_id"] == checkpoint_id

				# 2. READ - Get checkpoint
				retrieved = await saver.aget(config)
				assert retrieved is not None
				assert retrieved["id"] == checkpoint_id
				assert retrieved["channel_values"]["counter"] == 1
				assert retrieved["channel_values"]["messages"][0]["content"] == "Hello async world"

				# Also test get_tuple
				retrieved_tuple = await saver.aget_tuple(config)
				assert retrieved_tuple is not None
				assert retrieved_tuple.checkpoint["id"] == checkpoint_id
				assert retrieved_tuple.metadata["test_marker"] == test_marker
				assert retrieved_tuple.metadata["source"] == "async_crud_test"

				# 3. UPDATE - Update with new metadata
				updated_metadata = {
					"test_marker": test_marker,
					"source": "async_crud_test_updated",
					"step": 2,
				}

				checkpoint_id_2 = generate_unique_checkpoint_id()
				config_2 = generate_config_with_marker(
					thread_id=thread_id,
					checkpoint_id=checkpoint_id_2,
				)

				checkpoint_2, _ = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id_2,
					test_marker=test_marker,
					channel_values={
						"messages": [
							{"role": "user", "content": "Hello async world"},
							{"role": "assistant", "content": "Hi there!"},
						],
						"counter": 2,
					},
				)

				await saver.aput(config_2, checkpoint_2, updated_metadata, new_versions)

				# 4. LIST - Verify both checkpoints exist
				checkpoints = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}):
					checkpoints.append(cp)

				assert len(checkpoints) == 2
				checkpoint_ids = {cp.checkpoint["id"] for cp in checkpoints}
				assert checkpoint_id in checkpoint_ids
				assert checkpoint_id_2 in checkpoint_ids

				# 5. DELETE - Delete the thread
				await saver.adelete_thread(thread_id)

				# 6. VERIFY - Thread should be deleted
				deleted_checkpoints = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}):
					deleted_checkpoints.append(cp)

				assert len(deleted_checkpoints) == 0

				test_passed = True

			finally:
				if not test_passed:
					try:
						await saver.adelete_thread(thread_id)
					except:
						pass


@pytest.mark.asyncio
@pytest.mark.real_server_only
class TestAsyncIntegrationBinaryData:
	"""Test async binary data handling with real Go server."""

	async def test_async_binary_checkpoint_operations(self, http_base_url: str, http_api_key: str):
		"""Test async binary data storage and retrieval."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		# Create binary data
		binary_data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		checkpoint, metadata = generate_checkpoint_with_marker(
			checkpoint_id=checkpoint_id,
			test_marker=test_marker,
			channel_values={"binary": binary_data},
		)

		new_versions = {"binary": "1.0"}

		test_passed = False
		async with saver:
			try:
				# Store binary data
				await saver.aput(config, checkpoint, metadata, new_versions)

				# Retrieve and verify
				retrieved = await saver.aget(config)
				assert retrieved is not None
				assert retrieved["channel_values"]["binary"] == binary_data

				test_passed = True

			finally:
				if test_passed:
					await saver.adelete_thread(thread_id)

	async def test_async_large_binary_payloads(self, http_base_url: str, http_api_key: str):
		"""Test async handling of large binary payloads."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		# Create large binary data (1MB)
		large_binary = generate_binary_test_pattern(1024 * 1024)
		checksum = hashlib.sha256(large_binary).hexdigest()

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		checkpoint, metadata = generate_checkpoint_with_marker(
			checkpoint_id=checkpoint_id,
			test_marker=test_marker,
			channel_values={"large_data": large_binary},
		)

		new_versions = {"large_data": "1.0"}

		test_passed = False
		async with saver:
			try:
				# Store large binary data
				await saver.aput(config, checkpoint, metadata, new_versions)

				# Retrieve and verify integrity
				retrieved = await saver.aget(config)
				assert retrieved is not None

				retrieved_data = retrieved["channel_values"]["large_data"]
				retrieved_checksum = hashlib.sha256(retrieved_data).hexdigest()
				assert retrieved_checksum == checksum

				test_passed = True

			finally:
				if test_passed:
					await saver.adelete_thread(thread_id)


@pytest.mark.asyncio
@pytest.mark.real_server_only
class TestAsyncIntegrationMetadataFiltering:
	"""Test async metadata filtering with real Go server."""

	async def test_async_metadata_filtering(self, http_base_url: str, http_api_key: str):
		"""Test async filtering with complex metadata."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()

		test_cases = [
			{
				"checkpoint_id": generate_uuid_string(),
				"metadata": {"user": "alice", "type": "chat"},
			},
			{
				"checkpoint_id": generate_uuid_string(),
				"metadata": {"user": "bob", "type": "system"},
			},
			{
				"checkpoint_id": generate_uuid_string(),
				"metadata": {"user": "alice", "type": "chat", "nested": {"level": "high", "category": "important"}},
			},
		]

		async with saver:
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

				await saver.aput(config, checkpoint, case["metadata"], {})

			# Test simple filtering
			base_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

			alice_checkpoints = []
			async for cp in saver.alist(base_config, filter={"user": "alice"}):
				alice_checkpoints.append(cp)
			assert len(alice_checkpoints) == 2

			# Test nested filtering
			nested_checkpoints = []
			async for cp in saver.alist(base_config, filter={"nested": {"level": "high"}}):
				nested_checkpoints.append(cp)
			assert len(nested_checkpoints) == 1
			assert nested_checkpoints[0].metadata["nested"]["level"] == "high"

			# Cleanup
			await saver.adelete_thread(thread_id)


@pytest.mark.asyncio
@pytest.mark.real_server_only
class TestAsyncIntegrationErrorHandling:
	"""Test async error handling with real Go server."""

	async def test_async_not_found_handling(self, http_base_url: str, http_api_key: str):
		"""Test async 404 handling with real server."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		async with saver:
			# Try to get non-existent checkpoint
			non_existent_config = {
				"configurable": {
					"thread_id": generate_uuid_string(),
					"checkpoint_ns": "",
					"checkpoint_id": generate_uuid_string(),
				}
			}

			result = await saver.aget(non_existent_config)
			assert result is None

			tuple_result = await saver.aget_tuple(non_existent_config)
			assert tuple_result is None

	async def test_async_invalid_uuid_handling(self, http_base_url: str, http_api_key: str):
		"""Test async invalid UUID handling with real server."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		async with saver:
			# Invalid UUID should cause validation error
			invalid_config = {
				"configurable": {
					"thread_id": "not-a-uuid",
					"checkpoint_ns": "",
				}
			}

			# This should raise an HTTPClientError due to validation
			with pytest.raises(Exception):  # HTTPClientError or validation error
				await saver.aget(invalid_config)

	async def test_async_operation_timeout_handling(self, http_base_url: str, http_api_key: str):
		"""Test async handling of operation timeouts."""
		# Create saver with very short timeout
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
			timeout=0.001,  # 1ms timeout - extremely short
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		config = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id,
		)

		# Create large payload that might take time to process
		large_data = "x" * 1000000  # 1MB string
		checkpoint, metadata = generate_checkpoint_with_marker(
			checkpoint_id=checkpoint_id,
			test_marker=test_marker,
			channel_values={"large": large_data},
		)

		# Operations might timeout due to short timeout setting
		async with saver:
			try:
				# This might timeout
				await saver.aput(config, checkpoint, metadata, {})
			except (TimeoutException, asyncio.TimeoutError):
				# Timeout is expected with such short timeout
				pass
			except Exception as e:
				# Other timeout-related errors are also acceptable
				assert "timeout" in str(e).lower() or "time" in str(e).lower()

		# Create saver with reasonable timeout for cleanup
		cleanup_saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)
		async with cleanup_saver:
			try:
				await cleanup_saver.adelete_thread(thread_id)
			except:
				pass

	async def test_async_large_batch_stress(self, http_base_url: str, http_api_key: str):
		"""Test async handling of large batch operations."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		test_marker = generate_test_marker()
		checkpoint_ids = []

		test_passed = False
		try:
			async with saver:
				# Create many checkpoints concurrently
				num_checkpoints = 50
				tasks = []

				for i in range(num_checkpoints):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoint_ids.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						metadata={"batch": i, "total": num_checkpoints},
					)

					# Add to concurrent tasks
					tasks.append(saver.aput(config, checkpoint, metadata, {}))

				# Execute all concurrently
				results = await asyncio.gather(*tasks)
				assert all(r is not None for r in results)

				# List should handle large result sets
				checkpoints = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}):
					checkpoints.append(cp)

				assert len(checkpoints) == num_checkpoints

				# Verify all checkpoints are present
				retrieved_ids = {cp.checkpoint["id"] for cp in checkpoints}
				assert retrieved_ids == set(checkpoint_ids)

				test_passed = True
		finally:
			# Cleanup
			if not test_passed:
				async with saver:
					try:
						await saver.adelete_thread(thread_id)
					except:
						pass


@pytest.mark.asyncio
@pytest.mark.real_server_only
class TestAsyncRealServerConcurrency:
	"""Test async concurrent operations with real server."""

	async def test_async_concurrent_puts_different_threads(self, http_base_url: str, http_api_key: str):
		"""Test concurrent async PUT operations on different threads."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		# Create unique thread IDs for each concurrent operation
		num_threads = 5
		thread_ids = [generate_unique_thread_id() for _ in range(num_threads)]
		checkpoint_ids = [generate_unique_checkpoint_id() for _ in range(num_threads)]
		test_marker = generate_test_marker()

		test_passed = False
		try:
			async with saver:
				# Create tasks for concurrent operations
				tasks = []
				for i in range(num_threads):
					config = generate_config_with_marker(
						thread_id=thread_ids[i],
						checkpoint_id=checkpoint_ids[i],
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_ids[i],
						test_marker=test_marker,
						channel_values={"thread_index": i, "value": f"thread_{i}"},
					)

					tasks.append(saver.aput(config, checkpoint, metadata, {}))

				# Execute all concurrently
				results = await asyncio.gather(*tasks)

				# Verify all succeeded
				assert all(r is not None for r in results)

				# Verify each thread has its checkpoint
				for i in range(num_threads):
					config = generate_config_with_marker(
						thread_id=thread_ids[i],
						checkpoint_id=checkpoint_ids[i],
					)

					retrieved = await saver.aget(config)
					assert retrieved is not None
					assert retrieved["channel_values"]["thread_index"] == i
					assert retrieved["channel_values"]["value"] == f"thread_{i}"

				test_passed = True

		finally:
			if test_passed:
				# Cleanup all threads
				async with saver:
					cleanup_tasks = [saver.adelete_thread(tid) for tid in thread_ids]
					await asyncio.gather(*cleanup_tasks, return_exceptions=True)

	async def test_async_race_condition_handling(self, http_base_url: str, http_api_key: str):
		"""Test async race condition handling with concurrent updates."""
		saver = AsyncHTTPSingleStoreSaver(
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
		try:
			async with saver:
				# Create initial checkpoint
				initial_checkpoint, initial_metadata = generate_checkpoint_with_marker(
					checkpoint_id=checkpoint_id,
					test_marker=test_marker,
					channel_values={"value": "initial"},
				)

				await saver.aput(config, initial_checkpoint, initial_metadata, {})

				# Create concurrent update tasks
				update_count = 10
				update_tasks = []

				for i in range(update_count):
					updated_checkpoint = initial_checkpoint.copy()
					updated_checkpoint["channel_values"] = {"value": f"update_{i}"}

					update_tasks.append(saver.aput(config, updated_checkpoint, initial_metadata, {}))

				# Execute all updates concurrently
				results = await asyncio.gather(*update_tasks, return_exceptions=True)

				# Some updates might fail due to race conditions, but at least one should succeed
				successful_updates = [r for r in results if not isinstance(r, Exception)]
				assert len(successful_updates) > 0

				# Verify final state is from one of the updates
				final_checkpoint = await saver.aget(config)
				assert final_checkpoint is not None

				final_value = final_checkpoint["channel_values"]["value"]
				assert final_value.startswith("update_") or final_value == "initial"

				test_passed = True

		finally:
			if test_passed:
				async with saver:
					await saver.adelete_thread(thread_id)


@pytest.mark.asyncio
@pytest.mark.real_server_only
class TestAsyncRealServerDataIntegrity:
	"""Test async data integrity and consistency with real server."""

	async def test_async_checkpoint_immutability(self, http_base_url: str, http_api_key: str):
		"""Test that async checkpoints are immutable once created."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id = generate_unique_thread_id()
		checkpoint_id_1 = generate_unique_checkpoint_id()
		checkpoint_id_2 = generate_unique_checkpoint_id()
		test_marker = generate_test_marker()

		config_1 = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id_1,
		)

		config_2 = generate_config_with_marker(
			thread_id=thread_id,
			checkpoint_id=checkpoint_id_2,
		)

		checkpoint_1, metadata_1 = generate_checkpoint_with_marker(
			checkpoint_id=checkpoint_id_1,
			test_marker=test_marker,
			channel_values={"value": "original"},
		)

		checkpoint_2, metadata_2 = generate_checkpoint_with_marker(
			checkpoint_id=checkpoint_id_2,
			test_marker=test_marker,
			channel_values={"value": "updated"},
		)

		test_passed = False
		try:
			async with saver:
				# Create first checkpoint
				await saver.aput(config_1, checkpoint_1, metadata_1, {})

				# Create second checkpoint with same thread but different ID
				await saver.aput(config_2, checkpoint_2, metadata_2, {})

				# First checkpoint should remain unchanged
				retrieved_1 = await saver.aget(config_1)
				assert retrieved_1 is not None
				assert retrieved_1["channel_values"]["value"] == "original"

				# Second checkpoint should have new value
				retrieved_2 = await saver.aget(config_2)
				assert retrieved_2 is not None
				assert retrieved_2["channel_values"]["value"] == "updated"

				test_passed = True

		finally:
			if not test_passed:
				async with saver:
					try:
						await saver.adelete_thread(thread_id)
					except:
						pass

	async def test_async_thread_isolation(self, http_base_url: str, http_api_key: str):
		"""Test complete async isolation between different threads."""
		saver = AsyncHTTPSingleStoreSaver(
			base_url=http_base_url,
			api_key=http_api_key,
		)

		thread_id_1 = generate_unique_thread_id()
		thread_id_2 = generate_unique_thread_id()
		test_marker = generate_test_marker()

		# Create checkpoints for each thread
		checkpoints_thread_1 = []
		checkpoints_thread_2 = []

		test_passed = False
		try:
			async with saver:
				# Create checkpoints for thread 1
				for i in range(3):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoints_thread_1.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id_1,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={"thread": 1, "index": i},
						metadata={"thread_group": "group1"},
					)

					await saver.aput(config, checkpoint, metadata, {})

				# Create checkpoints for thread 2
				for i in range(3):
					checkpoint_id = generate_unique_checkpoint_id()
					checkpoints_thread_2.append(checkpoint_id)

					config = generate_config_with_marker(
						thread_id=thread_id_2,
						checkpoint_id=checkpoint_id,
					)

					checkpoint, metadata = generate_checkpoint_with_marker(
						checkpoint_id=checkpoint_id,
						test_marker=test_marker,
						channel_values={"thread": 2, "index": i},
						metadata={"thread_group": "group2"},
					)

					await saver.aput(config, checkpoint, metadata, {})

				# Verify thread 1 checkpoints are isolated
				thread_1_list = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id_1, "checkpoint_ns": ""}}):
					thread_1_list.append(cp)

				assert len(thread_1_list) == 3
				thread_1_ids = {cp.checkpoint["id"] for cp in thread_1_list}
				assert thread_1_ids == set(checkpoints_thread_1)

				# Verify thread 2 checkpoints are isolated
				thread_2_list = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id_2, "checkpoint_ns": ""}}):
					thread_2_list.append(cp)

				assert len(thread_2_list) == 3
				thread_2_ids = {cp.checkpoint["id"] for cp in thread_2_list}
				assert thread_2_ids == set(checkpoints_thread_2)

				# Verify no cross-contamination
				assert thread_1_ids.isdisjoint(thread_2_ids)

				# Delete thread 1
				await saver.adelete_thread(thread_id_1)

				# Thread 2 should be unaffected
				thread_2_list_after = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id_2, "checkpoint_ns": ""}}):
					thread_2_list_after.append(cp)

				assert len(thread_2_list_after) == 3

				# Thread 1 should be gone
				thread_1_list_after = []
				async for cp in saver.alist({"configurable": {"thread_id": thread_id_1, "checkpoint_ns": ""}}):
					thread_1_list_after.append(cp)

				assert len(thread_1_list_after) == 0

				test_passed = True

		finally:
			if not test_passed:
				async with saver:
					try:
						await saver.adelete_thread(thread_id_1)
						await saver.adelete_thread(thread_id_2)
					except:
						pass
