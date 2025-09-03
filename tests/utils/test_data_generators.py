"""Test data generators for real server integration tests.

This module provides utility functions for generating test data that works
with real servers containing persisted data. All generators create unique,
identifiable data using UUIDs and markers.
"""

import os
import uuid
import random
import string
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.singlestore.http.schemas import generate_uuid_string


def generate_unique_thread_id() -> str:
	"""Generate unique thread ID with test prefix for easy identification."""
	return str(uuid.uuid4())


def generate_unique_checkpoint_id() -> str:
	"""Generate unique checkpoint ID."""
	return str(uuid.uuid4())


def generate_unique_task_id() -> str:
	"""Generate unique task ID."""
	return str(uuid.uuid4())


def generate_test_marker() -> str:
	"""Generate unique marker for this test run.

	Use this marker in metadata to identify test-specific data
	among existing database records.
	"""
	return f"test_run_{uuid.uuid4().hex[:8]}"


def generate_timestamp() -> str:
	"""Generate ISO timestamp for current time."""
	return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_checkpoint_with_marker(
	checkpoint_id: Optional[str] = None,
	test_marker: Optional[str] = None,
	channel_values: Optional[Dict[str, Any]] = None,
	metadata: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
	"""Generate checkpoint with test marker for identification.

	Args:
	    checkpoint_id: Checkpoint ID, generates new UUID if None
	    test_marker: Test run marker, generates new one if None
	    channel_values: Channel values, empty dict if None
	    metadata: Additional metadata, merged with test marker

	Returns:
	    Checkpoint dictionary with test marker

	Note: Parent-child relationships are established via the config's checkpoint_id,
	    not stored in the checkpoint object itself.
	"""
	if checkpoint_id is None:
		checkpoint_id = generate_unique_checkpoint_id()
	if test_marker is None:
		test_marker = generate_test_marker()
	if channel_values is None:
		channel_values = {}

	# Build metadata with test marker
	final_metadata = {"test_marker": test_marker}
	if metadata:
		final_metadata.update(metadata)

	checkpoint = {
		"v": 1,
		"id": checkpoint_id,
		"ts": generate_timestamp(),
		"channel_values": channel_values,
		"channel_versions": {channel: "1.0" for channel in channel_values.keys()},
		"versions_seen": {},
		"pending_sends": [],
	}

	# Note: parent_checkpoint_id is NOT part of the Checkpoint TypedDict
	# Parent relationships are established via config["configurable"]["checkpoint_id"]

	return checkpoint, final_metadata


def generate_config_with_marker(
	thread_id: Optional[str] = None,
	checkpoint_id: Optional[str] = None,
	checkpoint_ns: str = "",
) -> Dict[str, Any]:
	"""Generate config with unique IDs.

	Args:
	    thread_id: Thread ID, generates new UUID if None
	    checkpoint_id: Checkpoint ID, omitted if None. Its the parent checkpoint id.
	    checkpoint_ns: Checkpoint namespace

	Returns:
	    Config dictionary
	"""
	if thread_id is None:
		thread_id = generate_unique_thread_id()

	config: Dict[str, Any] = {
		"configurable": {
			"thread_id": thread_id,
			"checkpoint_ns": checkpoint_ns,
		}
	}

	if checkpoint_id is not None:
		config["configurable"]["checkpoint_id"] = checkpoint_id

	return config


def generate_binary_test_pattern(size: int, pattern_type: str = "deadbeef") -> bytes:
	"""Generate recognizable binary pattern for testing.

	Args:
	    size: Size of binary data to generate
	    pattern_type: Type of pattern to generate
	        - "deadbeef": Repeating 0xDEADBEEF pattern
	        - "sequential": Sequential bytes 0x00, 0x01, 0x02...
	        - "random": Random bytes (but deterministic with seed)
	        - "edges": Edge case bytes (0x00, 0xFF, etc.)

	Returns:
	    Binary data of specified size
	"""
	if pattern_type == "deadbeef":
		pattern = b"\xde\xad\xbe\xef"
		return (pattern * (size // 4 + 1))[:size]
	elif pattern_type == "sequential":
		return bytes(i % 256 for i in range(size))
	elif pattern_type == "random":
		# Use deterministic random for reproducible tests
		random.seed(42)
		return bytes(random.randint(0, 255) for _ in range(size))
	elif pattern_type == "edges":
		edge_bytes = [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF]
		return bytes(edge_bytes[i % len(edge_bytes)] for i in range(size))
	else:
		raise ValueError(f"Unknown pattern_type: {pattern_type}")


def generate_unicode_test_strings() -> List[str]:
	"""Generate various Unicode test cases for string handling.

	Returns:
	    List of Unicode test strings covering edge cases
	"""
	return [
		"ASCII only text",
		"Émojis and symbols: 😀🎉🚀💯🔥⭐",
		"中文字符测试",
		"العربية النص",
		"עברית טקסט",
		"Русский текст",
		"Mathematical symbols: Ω≈ç√∫˜µ≤≥÷",
		"Zero-width chars: \u200b\u200c\u200d\u2060",
		"Combining chars: e\u0301a\u0300i\u0302",  # é à î
		"High code points: 𝕳𝖊𝖑𝖑𝖔 𝖂𝖔𝖗𝖑𝖉",
		"Control chars: \t\n\r",
		"Quotes and escapes: \"'\\`",
		# Note: Null bytes in metadata strings are not supported by the HTTP server
		# They get stripped during JSON processing in the Go server
		# "Null byte: \x00 in string",  # Commented out - server strips null bytes from metadata
	]


def generate_large_payload(size_mb: float) -> Dict[str, Any]:
	"""Generate large test payload of specified size.

	Args:
	    size_mb: Size in megabytes

	Returns:
	    Dictionary containing large data structure
	"""
	# Calculate approximate string size needed
	target_size = int(size_mb * 1024 * 1024)

	# Create base string pattern
	pattern = "Large payload test data " * 100
	repetitions = target_size // len(pattern) + 1
	large_string = pattern * repetitions

	return {
		"large_text": large_string[:target_size],
		"metadata": {
			"size_mb": size_mb,
			"pattern": "repeated text",
			"actual_size": len(large_string[:target_size]),
		},
	}


def generate_mixed_channel_types(test_marker: str) -> Dict[str, Any]:
	"""Generate channel values with mixed data types.

	Args:
	    test_marker: Test marker for identification

	Returns:
	    Dictionary with various data types
	"""
	return {
		"string_simple": "simple text value",
		"string_unicode": "Unicode: 你好世界 🌍",
		"integer": 42,
		"float": 3.14159,
		"boolean_true": True,
		"boolean_false": False,
		"null_value": None,
		"empty_string": "",
		"list_simple": [1, 2, 3],
		"list_mixed": [1, "two", 3.0, True, None],
		"list_nested": [[1, 2], [3, 4], [5, 6]],
		"dict_simple": {"key": "value"},
		"dict_nested": {
			"level1": {
				"level2": {
					"level3": "deep value",
					"test_marker": test_marker,
				}
			}
		},
		"binary_small": b"small binary \x00\x01\x02",
		"binary_large": generate_binary_test_pattern(1024, "deadbeef"),
	}


def generate_metadata_test_cases(test_marker: str) -> List[Dict[str, Any]]:
	"""Generate metadata test cases for filtering tests.

	Args:
	    test_marker: Test marker for identification

	Returns:
	    List of metadata dictionaries for different test scenarios
	"""
	return [
		# Simple filters
		{
			"test_marker": test_marker,
			"user": "alice",
			"type": "chat",
			"priority": 1,
		},
		{
			"test_marker": test_marker,
			"user": "bob",
			"type": "task",
			"priority": 2,
		},
		{
			"test_marker": test_marker,
			"user": "alice",
			"type": "chat",
			"priority": 3,
		},
		# Nested metadata
		{
			"test_marker": test_marker,
			"user": "charlie",
			"config": {
				"level": "high",
				"category": "important",
				"settings": {
					"auto_save": True,
					"notifications": False,
				},
			},
		},
		# Complex structures
		{
			"test_marker": test_marker,
			"tags": ["urgent", "customer", "bug"],
			"metrics": {
				"score": 9.5,
				"confidence": 0.87,
			},
			"history": [
				{"action": "created", "timestamp": "2024-01-01T00:00:00Z"},
				{"action": "updated", "timestamp": "2024-01-02T00:00:00Z"},
			],
		},
		# Edge cases
		{
			"test_marker": test_marker,
			"empty_dict": {},
			"empty_list": [],
			"null_value": None,
			"zero": 0,
			"empty_string": "",
			"unicode": "🚀 测试 العربية",
		},
	]


def generate_write_data_with_marker(test_marker: str) -> List[tuple]:
	"""Generate write data for checkpoint writes testing.

	Args:
	    test_marker: Test marker for identification

	Returns:
	    List of (channel, value) tuples
	"""
	return [
		("text_channel", f"Write test data - {test_marker}"),
		("number_channel", random.randint(1000, 9999)),
		("binary_channel", generate_binary_test_pattern(256, "deadbeef")),
		(
			"object_channel",
			{
				"test_marker": test_marker,
				"action": "write_test",
				"data": {"nested": "value"},
			},
		),
		("list_channel", [test_marker, "item2", 3, True]),
	]


def create_checkpoint_series(
	thread_id: str,
	count: int,
	test_marker: str,
	namespace: str = "",
) -> List[tuple]:
	"""Create a series of related checkpoints for testing.

	Args:
	    thread_id: Thread ID for all checkpoints
	    count: Number of checkpoints to create
	    test_marker: Test marker for identification
	    namespace: Checkpoint namespace

	Returns:
	    List of (config, checkpoint, metadata, versions) tuples
	"""
	checkpoints = []
	parent_id = None

	for i in range(count):
		checkpoint_id = generate_unique_checkpoint_id()

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
			channel_values={"step": i, "data": f"checkpoint_{i}"},
			metadata={"sequence": i, "step_name": f"step_{i}"},
		)

		versions = {"step": "1.0", "data": "1.0"}

		checkpoints.append((config, checkpoint, metadata, versions))
		parent_id = checkpoint_id

	return checkpoints


def generate_concurrent_test_data(test_marker: str, thread_count: int = 10) -> List[Dict[str, Any]]:
	"""Generate test data for concurrent operations testing.

	Args:
	    test_marker: Test marker for identification
	    thread_count: Number of threads to generate data for

	Returns:
	    List of test data dictionaries
	"""
	test_data = []

	for i in range(thread_count):
		thread_id = generate_unique_thread_id()
		checkpoint_id = generate_unique_checkpoint_id()

		test_data.append(
			{
				"thread_id": thread_id,
				"checkpoint_id": checkpoint_id,
				"config": {
					"configurable": {
						"thread_id": thread_id,
						"checkpoint_id": checkpoint_id,
						"checkpoint_ns": "",
					}
				},
				"checkpoint": {
					"v": 1,
					"id": checkpoint_id,
					"ts": generate_timestamp(),
					"channel_values": {
						"test_marker": test_marker,
						"thread_index": i,
						"data": f"concurrent_test_data_{i}",
					},
					"channel_versions": {"test_marker": "1.0", "thread_index": "1.0", "data": "1.0"},
					"versions_seen": {},
					"pending_sends": [],
				},
				"metadata": {
					"test_marker": test_marker,
					"thread_index": i,
					"test_type": "concurrent",
				},
				"versions": {"test_marker": "1.0", "thread_index": "1.0", "data": "1.0"},
			}
		)

	return test_data


def generate_invalid_uuid_test_cases() -> List[str]:
	"""Generate invalid UUID strings for error testing.

	Returns:
	    List of invalid UUID strings
	"""
	return [
		"not-a-uuid",
		"12345",
		"",
		"too-short",
		"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",  # Wrong format
		"123e4567-e89b-12d3-a456-42661417400",  # Too short
		"123e4567-e89b-12d3-a456-4266141740000",  # Too long
		"123e4567_e89b_12d3_a456_426614174000",  # Wrong separators
		"XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",  # Invalid hex
		None,  # None value
		123,  # Wrong type
	]


def cleanup_test_data(saver, test_marker: str, thread_ids: List[str] = None) -> None:
	"""Best-effort cleanup of test data.

	Args:
	    saver: HTTPSingleStoreSaver instance
	    test_marker: Test marker to identify data to clean
	    thread_ids: Optional list of specific thread IDs to clean

	Returns:
	    None
	"""

	try:
		if thread_ids:
			# Clear any existing client to force a new context
			saver._client = None

			# Clean specific threads - need to use client context
			with saver._get_client() as client:
				saver._client = client
				for thread_id in thread_ids:
					try:
						saver.delete_thread(thread_id)
					except Exception as e:
						raise e
	except Exception as e:
		raise e
