# type: ignore

from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
import singlestoredb
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS
from langgraph.checkpoint.singlestore import SingleStoreSaver
from tests.conftest import DEFAULT_SINGLESTORE_URI


def _exclude_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


@contextmanager
def _base_saver():
    """Fixture for regular connection mode testing."""
    database = f"test_{uuid4().hex[:16]}"
    # create unique db
    with singlestoredb.connect(DEFAULT_SINGLESTORE_URI, autocommit=True, results_type="dict") as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {database}")
    try:
        with singlestoredb.connect(
            f"{DEFAULT_SINGLESTORE_URI}/{database}",
            autocommit=True,
            results_type="dict",
        ) as conn:
            checkpointer = SingleStoreSaver(conn)
            checkpointer.setup()
            yield checkpointer
    finally:
        # drop unique db
        with singlestoredb.connect(DEFAULT_SINGLESTORE_URI, autocommit=True, results_type="dict") as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {database}")


@contextmanager
def _saver(name: str):
    if name == "base":
        with _base_saver() as saver:
            yield saver


@pytest.fixture
def test_data():
    """Fixture providing test data for checkpoint tests."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
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


def test_combined_metadata(test_data) -> None:
    with _saver("base") as saver:
        config = {
            "configurable": {
                "thread_id": "thread-2",
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
        saver.put(config, chkpnt, metadata, {})
        checkpoint = saver.get_tuple(config)
        assert checkpoint.metadata == {
            **metadata,
            "run_id": "my_run_id",
        }


def test_search(test_data) -> None:
    with _saver("base") as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})

        # call method / assertions
        query_1 = {"source": "input"}  # search by 1 key
        query_2 = {
            "step": 1,
            "writes": {"foo": "bar"},
        }  # search by multiple keys
        query_3: dict[str, Any] = {}  # search by no keys, return all checkpoints
        query_4 = {"source": "update", "step": 1}  # no match

        search_results_1 = list(saver.list(None, filter=query_1))
        assert len(search_results_1) == 1
        assert search_results_1[0].metadata == {
            **_exclude_keys(configs[0]["configurable"]),
            **metadata[0],
        }

        search_results_2 = list(saver.list(None, filter=query_2))
        assert len(search_results_2) == 1
        assert search_results_2[0].metadata == {
            **_exclude_keys(configs[1]["configurable"]),
            **metadata[1],
        }

        search_results_3 = list(saver.list(None, filter=query_3))
        assert len(search_results_3) == 3

        search_results_4 = list(saver.list(None, filter=query_4))
        assert len(search_results_4) == 0

        # search by config (defaults to checkpoints across all namespaces)
        search_results_5 = list(saver.list({"configurable": {"thread_id": "thread-2"}}))
        assert len(search_results_5) == 2
        assert {
            search_results_5[0].config["configurable"]["checkpoint_ns"],
            search_results_5[1].config["configurable"]["checkpoint_ns"],
        } == {"", "inner"}


def test_null_chars(test_data) -> None:
    with _saver("base") as saver:
        config = saver.put(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {"my_key": "\x00abc"},
            {},
        )
        assert saver.get_tuple(config).metadata["my_key"] == "abc"  # type: ignore
        assert (
            list(saver.list(None, filter={"my_key": "abc"}))[0].metadata["my_key"]
            == "abc"
        )


def test_pending_sends_migration() -> None:
    with _saver("base") as saver:
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # create the first checkpoint
        # and put some pending sends
        checkpoint_0 = empty_checkpoint()
        config = saver.put(config, checkpoint_0, {}, {})
        saver.put_writes(
            config, [(TASKS, "send-1"), (TASKS, "send-2")], task_id="task-1"
        )
        saver.put_writes(config, [(TASKS, "send-3")], task_id="task-2")

        # check that fetching checkpoint_0 doesn't attach pending sends
        # (they should be attached to the next checkpoint)
        tuple_0 = saver.get_tuple(config)
        assert tuple_0.checkpoint["channel_values"] == {}
        assert tuple_0.checkpoint["channel_versions"] == {}

        # create the second checkpoint
        checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
        config = saver.put(config, checkpoint_1, {}, {})

        # check that pending sends are attached to checkpoint_1
        checkpoint_1 = saver.get_tuple(config)
        assert checkpoint_1.checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in checkpoint_1.checkpoint["channel_versions"]

        # check that list also applies the migration
        search_results = [
            c for c in saver.list({"configurable": {"thread_id": "thread-1"}})
        ]
        assert len(search_results) == 2
        assert search_results[-1].checkpoint["channel_values"] == {}
        assert search_results[-1].checkpoint["channel_versions"] == {}
        assert search_results[0].checkpoint["channel_values"] == {
            TASKS: ["send-1", "send-2", "send-3"]
        }
        assert TASKS in search_results[0].checkpoint["channel_versions"]


def test_basic_get_put() -> None:
    """Test basic get and put operations."""
    with _base_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # Test put
        checkpoint = empty_checkpoint()
        metadata = {"test": "value"}
        result_config = saver.put(config, checkpoint, metadata, {})

        # Test get
        retrieved = saver.get_tuple(result_config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint["id"]
        assert retrieved.metadata["test"] == "value"


def test_get_nonexistent() -> None:
    """Test getting a non-existent checkpoint."""
    with _base_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "nonexistent",
                "checkpoint_ns": "",
            }
        }

        result = saver.get_tuple(config)
        assert result is None


def test_list_empty() -> None:
    """Test listing when no checkpoints exist."""
    with _base_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "empty",
                "checkpoint_ns": "",
            }
        }

        results = list(saver.list(config))
        assert len(results) == 0


def test_delete_thread() -> None:
    """Test deleting all checkpoints for a thread."""
    with _base_saver() as saver:
        config = {
            "configurable": {
                "thread_id": "delete-test",
                "checkpoint_ns": "",
            }
        }

        # Add some checkpoints
        checkpoint1 = empty_checkpoint()
        checkpoint2 = create_checkpoint(checkpoint1, {}, 1)

        saver.put(config, checkpoint1, {"step": 1}, {})
        saver.put(config, checkpoint2, {"step": 2}, {})

        # Verify they exist
        results = list(saver.list(config))
        assert len(results) == 2

        # Delete the thread
        saver.delete_thread("delete-test")

        # Verify they're gone
        results = list(saver.list(config))
        assert len(results) == 0
