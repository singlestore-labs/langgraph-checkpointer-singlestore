# type: ignore
"""Shared test utilities for checkpoint tests."""

from typing import Any

from langgraph.checkpoint.base import (
	EXCLUDED_METADATA_KEYS,
	CheckpointMetadata,
	create_checkpoint,
	empty_checkpoint,
)


def exclude_private_keys(config: dict[str, Any]) -> dict[str, Any]:
	"""Exclude private keys from metadata."""
	return {k: v for k, v in config.items() if k not in EXCLUDED_METADATA_KEYS}


def create_test_checkpoints() -> list[dict[str, Any]]:
	"""Create standard test checkpoint data for search tests."""
	return [
		{
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "cp-1",
			"parent_checkpoint_id": None,
			"checkpoint": create_checkpoint(empty_checkpoint(), {}, 1),
			"metadata": {"source": "input", "step": 0},
			"channel_values": [],
			"pending_writes": [],
		},
		{
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "cp-2",
			"parent_checkpoint_id": "cp-1",
			"checkpoint": create_checkpoint(empty_checkpoint(), {}, 2),
			"metadata": {"source": "loop", "step": 1, "writes": {"foo": "bar"}},
			"channel_values": [],
			"pending_writes": [],
		},
		{
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "cp-3",
			"parent_checkpoint_id": "cp-2",
			"checkpoint": create_checkpoint(empty_checkpoint(), {}, 3),
			"metadata": {"source": "loop", "step": 2, "writes": None},
			"channel_values": [],
			"pending_writes": [],
		},
	]


def filter_checkpoints(checkpoints: list[dict[str, Any]], filter_query: dict[str, Any]) -> list[dict[str, Any]]:
	"""Filter checkpoints based on metadata query."""
	if not filter_query:
		return checkpoints

	filtered = []
	for cp in checkpoints:
		match = all(cp["metadata"].get(k) == v for k, v in filter_query.items())
		if match:
			filtered.append(cp)
	return filtered


def create_large_metadata(num_keys: int = 100) -> dict[str, str]:
	"""Create large metadata payload for testing."""
	return {f"key_{i}": f"value_{i}" * 100 for i in range(num_keys)}


def create_unicode_metadata() -> dict[str, str]:
	"""Create metadata with various Unicode characters."""
	return {
		"emoji": "🎉🚀💡",
		"chinese": "你好世界",
		"arabic": "مرحبا بالعالم",
		"special": "Ñoño™€",
		"russian": "Привет мир",
		"japanese": "こんにちは世界",
		"korean": "안녕하세요",
		"hebrew": "שלום עולם",
	}


def create_metadata_with_private_keys() -> dict[str, Any]:
	"""Create metadata with private keys that should be excluded."""
	return {
		"source": "loop",
		"step": 1,
		"writes": {"foo": "bar"},
		"score": None,
		"__private_key": "should_be_excluded",
		"__super_private_key": "super_private_value",
	}


def create_empty_checkpoint() -> dict[str, Any]:
	"""Create checkpoint with empty values for edge case testing."""
	return {
		"v": 1,
		"id": "checkpoint-empty",
		"ts": "2024-01-01T00:00:00Z",
		"channel_values": {},
		"channel_versions": {},
		"versions_seen": {},
		"pending_sends": [],
	}


def create_checkpoint_with_binary_data() -> tuple[dict[str, Any], bytes]:
	"""Create checkpoint with binary data for blob testing."""
	binary_data = b"This is binary data \x00\x01\x02\x03\x04"
	checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
	checkpoint["channel_values"]["binary_channel"] = binary_data
	return checkpoint, binary_data


def create_search_test_queries() -> list[tuple[dict[str, Any], int]]:
	"""Create standard search queries with expected result counts."""
	return [
		({"source": "input"}, 1),  # Single key filter
		({"step": 1, "writes": {"foo": "bar"}}, 1),  # Multiple keys
		({}, 3),  # No filter - return all
		({"source": "update", "step": 1}, 0),  # No match
	]


def assert_checkpoint_metadata(
	checkpoint_metadata: dict[str, Any],
	expected_metadata: dict[str, Any],
	exclude_private: bool = True,
) -> None:
	"""Assert checkpoint metadata matches expected, optionally excluding private keys."""
	if exclude_private:
		actual = exclude_private_keys(checkpoint_metadata)
		expected = exclude_private_keys(expected_metadata)
	else:
		actual = checkpoint_metadata
		expected = expected_metadata

	assert actual == expected


def create_config_with_metadata(
	thread_id: str,
	checkpoint_ns: str = "",
	metadata: dict[str, Any] | None = None,
	private_keys: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Create a config dict with metadata and optional private keys."""
	config = {
		"configurable": {
			"thread_id": thread_id,
			"checkpoint_ns": checkpoint_ns,
		}
	}

	if private_keys:
		config["configurable"].update(private_keys)

	if metadata:
		config["metadata"] = metadata

	return config
