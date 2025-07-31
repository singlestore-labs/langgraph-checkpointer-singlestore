# type: ignore

from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
	EXCLUDED_METADATA_KEYS,
	Checkpoint,
	CheckpointMetadata,
	create_checkpoint,
	empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
	return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


class TestAsyncCheckpoint:
	"""Test class for async checkpoint operations."""

	@pytest.mark.asyncio
	async def test_combined_metadata(self, test_data, async_saver) -> None:
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"thread-2-{unique_id}",
				"checkpoint_ns": "",
				"__super_private_key": "super_private_value",
			},
			"metadata": {"run_id": "my_run_id"},
		}
		chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
		metadata: CheckpointMetadata = {
			"source": "loop",
			"step": 1,
			"writes": {"foo": "bar"},
			"score": None,
		}
		await async_saver.aput(config, chkpnt, metadata, {})
		checkpoint = await async_saver.aget_tuple(config)
		assert checkpoint.metadata == {
			**metadata,
			"run_id": "my_run_id",
		}

	@pytest.mark.asyncio
	async def test_asearch(self, test_data, async_saver) -> None:
		configs = test_data["configs"]
		checkpoints = test_data["checkpoints"]
		metadata = test_data["metadata"]

		await async_saver.aput(configs[0], checkpoints[0], metadata[0], {})
		await async_saver.aput(configs[1], checkpoints[1], metadata[1], {})
		await async_saver.aput(configs[2], checkpoints[2], metadata[2], {})

		# call method / assertions
		query_1 = {"source": "input"}  # search by 1 key
		query_2 = {
			"step": 1,
			"writes": {"foo": "bar"},
		}  # search by multiple keys
		query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
		query_4 = {"source": "update", "step": 1}  # no match

		search_results_1 = [c async for c in async_saver.alist(None, filter=query_1)]
		assert len(search_results_1) == 1
		assert search_results_1[0].metadata == {
			**_exclude_keys(configs[0]["configurable"]),
			**metadata[0],
		}

		search_results_2 = [c async for c in async_saver.alist(None, filter=query_2)]
		assert len(search_results_2) == 1
		assert search_results_2[0].metadata == {
			**_exclude_keys(configs[1]["configurable"]),
			**metadata[1],
		}

		search_results_3 = [c async for c in async_saver.alist(None, filter=query_3)]
		assert len(search_results_3) == 3

		search_results_4 = [c async for c in async_saver.alist(None, filter=query_4)]
		assert len(search_results_4) == 0

		# search by config (defaults to checkpoints across all namespaces)
		# Use the thread_id from the test_data configs
		thread_id = configs[1]["configurable"]["thread_id"]
		search_results_5 = [c async for c in async_saver.alist({"configurable": {"thread_id": thread_id}})]
		assert len(search_results_5) == 2
		assert {
			search_results_5[0].config["configurable"]["checkpoint_ns"],
			search_results_5[1].config["configurable"]["checkpoint_ns"],
		} == {"", "inner"}

	@pytest.mark.asyncio
	async def test_null_chars(self, test_data, async_saver) -> None:
		config = await async_saver.aput(
			test_data["configs"][0],
			test_data["checkpoints"][0],
			{"my_key": "\x00abc"},
			{},
		)
		assert (await async_saver.aget_tuple(config)).metadata["my_key"] == "abc"  # type: ignore
		assert [c async for c in async_saver.alist(None, filter={"my_key": "abc"})][0].metadata["my_key"] == "abc"

	@pytest.mark.asyncio
	async def test_pending_sends_migration(self, async_saver) -> None:
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"thread-1-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		# create the first checkpoint
		# and put some pending sends
		checkpoint_0 = empty_checkpoint()
		config = await async_saver.aput(config, checkpoint_0, {}, {})
		await async_saver.aput_writes(config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1")
		await async_saver.aput_writes(config, [(TASKS, "send-3")], task_id="task-2")

		# check that fetching checkpoint_0 doesn't attach pending sends
		# (they should be attached to the next checkpoint)
		tuple_0 = await async_saver.aget_tuple(config)
		assert tuple_0.checkpoint["channel_values"] == {}
		assert tuple_0.checkpoint["channel_versions"] == {}

		# create the second checkpoint
		checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
		config = await async_saver.aput(config, checkpoint_1, {}, {})

		# check that pending sends are attached to checkpoint_1
		tuple_1 = await async_saver.aget_tuple(config)
		assert tuple_1.checkpoint["channel_values"] == {TASKS: ["send-1", "send-2", "send-3"]}
		assert TASKS in tuple_1.checkpoint["channel_versions"]

		# check that list also applies the migration
		search_results = [c async for c in async_saver.alist({"configurable": {"thread_id": f"thread-1-{unique_id}"}})]
		assert len(search_results) == 2
		assert search_results[-1].checkpoint["channel_values"] == {}
		assert search_results[-1].checkpoint["channel_versions"] == {}
		assert search_results[0].checkpoint["channel_values"] == {TASKS: ["send-1", "send-2", "send-3"]}
		assert TASKS in search_results[0].checkpoint["channel_versions"]

	@pytest.mark.asyncio
	async def test_basic_async_get_put(self, async_saver) -> None:
		"""Test basic async get and put operations."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"thread-1-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		# Test put
		checkpoint = empty_checkpoint()
		metadata = {"test": "value"}
		result_config = await async_saver.aput(config, checkpoint, metadata, {})

		# Test get
		retrieved = await async_saver.aget_tuple(result_config)
		assert retrieved is not None
		assert retrieved.checkpoint["id"] == checkpoint["id"]
		assert retrieved.metadata["test"] == "value"

	@pytest.mark.asyncio
	async def test_async_get_nonexistent(self, async_saver) -> None:
		"""Test getting a non-existent checkpoint asynchronously."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"nonexistent-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		result = await async_saver.aget_tuple(config)
		assert result is None

	@pytest.mark.asyncio
	async def test_async_list_empty(self, async_saver) -> None:
		"""Test listing when no checkpoints exist asynchronously."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"empty-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		results = [c async for c in async_saver.alist(config)]
		assert len(results) == 0

	@pytest.mark.asyncio
	async def test_async_delete_thread(self, async_saver) -> None:
		"""Test deleting all checkpoints for a thread asynchronously."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"delete-test-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		# Add some checkpoints
		checkpoint1 = empty_checkpoint()
		checkpoint2 = create_checkpoint(checkpoint1, {}, 1)

		await async_saver.aput(config, checkpoint1, {"step": 1}, {})
		await async_saver.aput(config, checkpoint2, {"step": 2}, {})

		# Verify they exist
		results = [c async for c in async_saver.alist(config)]
		assert len(results) == 2

		# Delete the thread
		await async_saver.adelete_thread(f"delete-test-{unique_id}")

		# Verify they're gone
		results = [c async for c in async_saver.alist(config)]
		assert len(results) == 0

	@pytest.mark.asyncio
	async def test_from_conn_string(self) -> None:
		"""Test creating AsyncSingleStoreSaver from connection string."""
		from langgraph.checkpoint.singlestore.aio import AsyncSingleStoreSaver
		from tests.conftest import DEFAULT_URI_WITHOUT_DB
		import singlestoredb
		from uuid import uuid4

		database = f"test_{uuid4().hex[:16]}"

		# Create the database first
		with singlestoredb.connect(DEFAULT_URI_WITHOUT_DB, autocommit=True, results_type="dict") as conn:
			with conn.cursor() as cursor:
				cursor.execute(f"CREATE DATABASE {database}")

		try:
			conn_string = f"{DEFAULT_URI_WITHOUT_DB}/{database}"
			async with AsyncSingleStoreSaver.from_conn_string(conn_string) as saver:
				await saver.setup()

				# Test basic functionality
				config = {
					"configurable": {
						"thread_id": "test-thread",
						"checkpoint_ns": "",
					}
				}

				checkpoint = empty_checkpoint()
				result_config = await saver.aput(config, checkpoint, {"test": "data"}, {})
				retrieved = await saver.aget_tuple(result_config)

				assert retrieved is not None
				assert retrieved.metadata["test"] == "data"
		finally:
			# Clean up
			with singlestoredb.connect(DEFAULT_URI_WITHOUT_DB, autocommit=True, results_type="dict") as conn:
				with conn.cursor() as cursor:
					cursor.execute(f"DROP DATABASE {database}")

	@pytest.mark.asyncio
	async def test_async_get_method(self, async_saver) -> None:
		"""Test the aget method (not aget_tuple)."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"test-get-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		# Test with non-existent checkpoint
		checkpoint = await async_saver.aget(config)
		assert checkpoint is None

		# Add a checkpoint
		test_checkpoint = empty_checkpoint()
		result_config = await async_saver.aput(config, test_checkpoint, {"test": "value"}, {})

		# Test with existing checkpoint
		retrieved_checkpoint = await async_saver.aget(result_config)
		assert retrieved_checkpoint is not None
		assert retrieved_checkpoint["id"] == test_checkpoint["id"]
