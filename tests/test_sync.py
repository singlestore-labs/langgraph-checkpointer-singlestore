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


class TestSyncCheckpoint:
	"""Test class for sync checkpoint operations."""

	def test_combined_metadata(self, test_data, sync_saver) -> None:
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
		sync_saver.put(config, chkpnt, metadata, {})
		checkpoint = sync_saver.get_tuple(config)
		assert checkpoint.metadata == {
			**metadata,
			"run_id": "my_run_id",
		}

	def test_search(self, test_data, sync_saver) -> None:
		configs = test_data["configs"]
		checkpoints = test_data["checkpoints"]
		metadata = test_data["metadata"]

		sync_saver.put(configs[0], checkpoints[0], metadata[0], {})
		sync_saver.put(configs[1], checkpoints[1], metadata[1], {})
		sync_saver.put(configs[2], checkpoints[2], metadata[2], {})

		# call method / assertions
		query_1 = {"source": "input"}  # search by 1 key
		query_2 = {
			"step": 1,
			"writes": {"foo": "bar"},
		}  # search by multiple keys
		query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
		query_4 = {"source": "update", "step": 1}  # no match

		search_results_1 = list(sync_saver.list(None, filter=query_1))
		assert len(search_results_1) == 1
		assert search_results_1[0].metadata == {
			**_exclude_keys(configs[0]["configurable"]),
			**metadata[0],
		}

		search_results_2 = list(sync_saver.list(None, filter=query_2))
		assert len(search_results_2) == 1
		assert search_results_2[0].metadata == {
			**_exclude_keys(configs[1]["configurable"]),
			**metadata[1],
		}

		search_results_3 = list(sync_saver.list(None, filter=query_3))
		assert len(search_results_3) == 3

		search_results_4 = list(sync_saver.list(None, filter=query_4))
		assert len(search_results_4) == 0

		# search by config (defaults to checkpoints across all namespaces)
		# Use the thread_id from the test_data configs
		thread_id = configs[1]["configurable"]["thread_id"]
		search_results_5 = list(sync_saver.list({"configurable": {"thread_id": thread_id}}))
		assert len(search_results_5) == 2
		assert {
			search_results_5[0].config["configurable"]["checkpoint_ns"],
			search_results_5[1].config["configurable"]["checkpoint_ns"],
		} == {"", "inner"}

	def test_null_chars(self, test_data, sync_saver) -> None:
		config = sync_saver.put(
			test_data["configs"][0],
			test_data["checkpoints"][0],
			{"my_key": "\x00abc"},
			{},
		)
		assert sync_saver.get_tuple(config).metadata["my_key"] == "abc"  # type: ignore
		assert list(sync_saver.list(None, filter={"my_key": "abc"}))[0].metadata["my_key"] == "abc"

	def test_pending_sends_migration(self, sync_saver) -> None:
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
		config = sync_saver.put(config, checkpoint_0, {}, {})
		sync_saver.put_writes(config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1")
		sync_saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")

		# check that fetching checkpoint_0 doesn't attach pending sends
		# (they should be attached to the next checkpoint)
		tuple_0 = sync_saver.get_tuple(config)
		assert tuple_0.checkpoint["channel_values"] == {}
		assert tuple_0.checkpoint["channel_versions"] == {}

		# create the second checkpoint
		checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
		config = sync_saver.put(config, checkpoint_1, {}, {})

		# check that pending sends are attached to checkpoint_1
		checkpoint_1 = sync_saver.get_tuple(config)
		assert checkpoint_1.checkpoint["channel_values"] == {TASKS: ["send-1", "send-2", "send-3"]}
		assert TASKS in checkpoint_1.checkpoint["channel_versions"]

		# check that list also applies the migration
		search_results = [c for c in sync_saver.list({"configurable": {"thread_id": f"thread-1-{unique_id}"}})]
		assert len(search_results) == 2
		assert search_results[-1].checkpoint["channel_values"] == {}
		assert search_results[-1].checkpoint["channel_versions"] == {}
		assert search_results[0].checkpoint["channel_values"] == {TASKS: ["send-1", "send-2", "send-3"]}
		assert TASKS in search_results[0].checkpoint["channel_versions"]

	def test_basic_get_put(self, sync_saver) -> None:
		"""Test basic get and put operations."""
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
		result_config = sync_saver.put(config, checkpoint, metadata, {})

		# Test get
		retrieved = sync_saver.get_tuple(result_config)
		assert retrieved is not None
		assert retrieved.checkpoint["id"] == checkpoint["id"]
		assert retrieved.metadata["test"] == "value"

	def test_get_nonexistent(self, sync_saver) -> None:
		"""Test getting a non-existent checkpoint."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"nonexistent-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		result = sync_saver.get_tuple(config)
		assert result is None

	def test_list_empty(self, sync_saver) -> None:
		"""Test listing when no checkpoints exist."""
		import uuid

		unique_id = uuid.uuid4().hex[:8]

		config = {
			"configurable": {
				"thread_id": f"empty-{unique_id}",
				"checkpoint_ns": "",
			}
		}

		results = list(sync_saver.list(config))
		assert len(results) == 0

	def test_delete_thread(self, sync_saver) -> None:
		"""Test deleting all checkpoints for a thread."""
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

		sync_saver.put(config, checkpoint1, {"step": 1}, {})
		sync_saver.put(config, checkpoint2, {"step": 2}, {})

		# Verify they exist
		results = list(sync_saver.list(config))
		assert len(results) == 2

		# Delete the thread
		sync_saver.delete_thread(f"delete-test-{unique_id}")

		# Verify they're gone
		results = list(sync_saver.list(config))
		assert len(results) == 0
