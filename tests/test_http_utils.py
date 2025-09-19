"""Tests for HTTP utility functions."""

from __future__ import annotations

import base64
import uuid
from typing import Any

import pytest

from langgraph.checkpoint.singlestore.http.utils import (
	decode_from_base64,
	encode_to_base64,
	prepare_metadata_filter,
	transform_channel_values,
	transform_pending_writes,
)


class TestBase64Functions:
	"""Test base64 encoding and decoding functions."""

	def test_encode_to_base64_simple(self):
		"""Test encoding simple binary data to base64."""
		data = b"Hello, World!"
		result = encode_to_base64(data)
		assert result == "SGVsbG8sIFdvcmxkIQ=="
		assert isinstance(result, str)

	def test_encode_to_base64_empty(self):
		"""Test encoding empty binary data."""
		data = b""
		result = encode_to_base64(data)
		assert result == ""

	def test_encode_to_base64_binary(self):
		"""Test encoding binary data with non-ASCII bytes."""
		data = b"\x00\x01\x02\x03\x04\x05"
		result = encode_to_base64(data)
		assert result == "AAECAwQF"

	def test_encode_to_base64_unicode_bytes(self):
		"""Test encoding UTF-8 encoded Unicode text."""
		text = "Hello 世界 🌍"
		data = text.encode("utf-8")
		result = encode_to_base64(data)
		# Verify we can decode it back
		decoded = base64.b64decode(result).decode("utf-8")
		assert decoded == text

	def test_decode_from_base64_simple(self):
		"""Test decoding simple base64 string."""
		data = "SGVsbG8sIFdvcmxkIQ=="
		result = decode_from_base64(data)
		assert result == b"Hello, World!"
		assert isinstance(result, bytes)

	def test_decode_from_base64_empty(self):
		"""Test decoding empty base64 string."""
		data = ""
		result = decode_from_base64(data)
		assert result == b""

	def test_decode_from_base64_binary(self):
		"""Test decoding base64 with binary data."""
		data = "AAECAwQF"
		result = decode_from_base64(data)
		assert result == b"\x00\x01\x02\x03\x04\x05"

	def test_encode_decode_roundtrip(self):
		"""Test that encode and decode are inverse operations."""
		original = b"Test data with special chars: !@#$%^&*()"
		encoded = encode_to_base64(original)
		decoded = decode_from_base64(encoded)
		assert decoded == original

	def test_encode_decode_large_data(self):
		"""Test encoding and decoding large binary data."""
		# Create 1MB of random-like data
		original = bytes(range(256)) * 4096
		encoded = encode_to_base64(original)
		decoded = decode_from_base64(encoded)
		assert decoded == original
		assert len(decoded) == 1048576  # 1MB

	def test_decode_invalid_base64(self):
		"""Test decoding invalid base64 strings."""
		with pytest.raises(Exception):  # base64.b64decode raises binascii.Error
			decode_from_base64("This is not base64!")

	def test_decode_base64_with_padding(self):
		"""Test decoding base64 with different padding scenarios."""
		# No padding needed
		assert decode_from_base64("YWJj") == b"abc"
		# One padding character
		assert decode_from_base64("YWI=") == b"ab"
		# Two padding characters
		assert decode_from_base64("YQ==") == b"a"


class TestTransformChannelValues:
	"""Test transform_channel_values function."""

	def test_transform_channel_values_simple(self):
		"""Test transforming simple channel values."""
		input_data = [
			["channel1", "json", "SGVsbG8="],  # "Hello" in base64
			["channel2", "text", "V29ybGQ="],  # "World" in base64
		]
		result = transform_channel_values(input_data)

		assert len(result) == 2
		assert result[0] == (b"channel1", b"json", b"Hello")
		assert result[1] == (b"channel2", b"text", b"World")

	def test_transform_channel_values_empty_list(self):
		"""Test transforming empty channel values list."""
		result = transform_channel_values([])
		assert result is None

	def test_transform_channel_values_none(self):
		"""Test transforming None channel values."""
		result = transform_channel_values(None)
		assert result is None

	def test_transform_channel_values_empty_blob(self):
		"""Test transforming channel values with empty blob."""
		input_data = [
			["channel1", "json", ""],
			["channel2", "text", None],
		]
		result = transform_channel_values(input_data)

		assert len(result) == 2
		assert result[0] == (b"channel1", b"json", b"")
		assert result[1] == (b"channel2", b"text", b"")

	def test_transform_channel_values_unicode(self):
		"""Test transforming channel values with Unicode names."""
		input_data = [
			["channel_😀", "json", "SGVsbG8="],
			["频道_中文", "text", "V29ybGQ="],
		]
		result = transform_channel_values(input_data)

		assert len(result) == 2
		assert result[0][0] == "channel_😀".encode("utf-8")
		assert result[1][0] == "频道_中文".encode("utf-8")

	def test_transform_channel_values_short_item(self):
		"""Test transforming channel values with items shorter than expected."""
		input_data = [
			["channel1", "json"],  # Missing blob
			["channel2"],  # Missing type and blob
		]
		result = transform_channel_values(input_data)

		# Should skip items with less than 3 elements
		assert result is None or len(result) == 0

	def test_transform_channel_values_extra_fields(self):
		"""Test transforming channel values with extra fields."""
		input_data = [
			["channel1", "json", "SGVsbG8=", "extra", "fields"],
		]
		result = transform_channel_values(input_data)

		# Should only use first 3 fields
		assert len(result) == 1
		assert result[0] == (b"channel1", b"json", b"Hello")

	def test_transform_channel_values_binary_data(self):
		"""Test transforming channel values with binary data."""
		binary = b"\x00\x01\x02\x03\x04"
		encoded = base64.b64encode(binary).decode("utf-8")
		input_data = [
			["binary_channel", "bytes", encoded],
		]
		result = transform_channel_values(input_data)

		assert len(result) == 1
		assert result[0] == (b"binary_channel", b"bytes", binary)


class TestTransformPendingWrites:
	"""Test transform_pending_writes function."""

	def test_transform_pending_writes_simple(self):
		"""Test transforming simple pending writes."""
		input_data = [
			["task1", "channel1", "json", "SGVsbG8="],
			["task2", "channel2", "text", "V29ybGQ="],
		]
		result = transform_pending_writes(input_data)

		assert len(result) == 2
		assert result[0] == (b"task1", b"channel1", b"json", b"Hello")
		assert result[1] == (b"task2", b"channel2", b"text", b"World")

	def test_transform_pending_writes_empty_list(self):
		"""Test transforming empty pending writes list."""
		result = transform_pending_writes([])
		assert result is None

	def test_transform_pending_writes_none(self):
		"""Test transforming None pending writes."""
		result = transform_pending_writes(None)
		assert result is None

	def test_transform_pending_writes_empty_blob(self):
		"""Test transforming pending writes with empty blob."""
		input_data = [
			["task1", "channel1", "json", ""],
			["task2", "channel2", "text", None],
		]
		result = transform_pending_writes(input_data)

		assert len(result) == 2
		assert result[0] == (b"task1", b"channel1", b"json", b"")
		assert result[1] == (b"task2", b"channel2", b"text", b"")

	def test_transform_pending_writes_unicode(self):
		"""Test transforming pending writes with Unicode task IDs."""
		input_data = [
			["task_😀", "channel1", "json", "SGVsbG8="],
			["任务_中文", "channel2", "text", "V29ybGQ="],
		]
		result = transform_pending_writes(input_data)

		assert len(result) == 2
		assert result[0][0] == "task_😀".encode("utf-8")
		assert result[1][0] == "任务_中文".encode("utf-8")

	def test_transform_pending_writes_short_item(self):
		"""Test transforming pending writes with items shorter than expected."""
		input_data = [
			["task1", "channel1", "json"],  # Missing blob
			["task2", "channel2"],  # Missing type and blob
			["task3"],  # Missing channel, type, and blob
		]
		result = transform_pending_writes(input_data)

		# Should skip items with less than 4 elements
		assert result is None or len(result) == 0

	def test_transform_pending_writes_extra_fields(self):
		"""Test transforming pending writes with extra fields."""
		input_data = [
			["task1", "channel1", "json", "SGVsbG8=", "extra", "fields"],
		]
		result = transform_pending_writes(input_data)

		# Should only use first 4 fields
		assert len(result) == 1
		assert result[0] == (b"task1", b"channel1", b"json", b"Hello")

	def test_transform_pending_writes_complex_task_paths(self):
		"""Test transforming pending writes with complex task paths."""
		input_data = [
			["task/subtask/1", "channel1", "json", "SGVsbG8="],
			["task.subtask.2", "channel2", "text", "V29ybGQ="],
		]
		result = transform_pending_writes(input_data)

		assert len(result) == 2
		assert result[0][0] == b"task/subtask/1"
		assert result[1][0] == b"task.subtask.2"


class TestPrepareMetadataFilter:
	"""Test prepare_metadata_filter function."""

	def test_prepare_metadata_filter_simple(self):
		"""Test preparing simple metadata filter."""
		metadata = {
			"key1": "value1",
			"key2": 123,
			"key3": True,
		}
		result = prepare_metadata_filter(metadata)

		assert result == {
			"key1": "value1",
			"key2": 123,
			"key3": True,
		}

	def test_prepare_metadata_filter_nested_dict(self):
		"""Test preparing metadata filter with nested dictionaries."""
		metadata = {
			"simple": "value",
			"nested": {"level1": {"level2": "deep_value"}},
		}
		result = prepare_metadata_filter(metadata)

		assert result["simple"] == "value"
		assert result["nested"] == {"level1": {"level2": "deep_value"}}

	def test_prepare_metadata_filter_list(self):
		"""Test preparing metadata filter with lists."""
		metadata = {
			"tags": ["tag1", "tag2", "tag3"],
			"numbers": [1, 2, 3],
			"mixed": ["text", 123, True],
		}
		result = prepare_metadata_filter(metadata)

		assert result["tags"] == ["tag1", "tag2", "tag3"]
		assert result["numbers"] == [1, 2, 3]
		assert result["mixed"] == ["text", 123, True]

	def test_prepare_metadata_filter_uuid(self):
		"""Test preparing metadata filter with UUID values."""
		test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
		metadata = {
			"id": test_uuid,
			"name": "test",
		}
		result = prepare_metadata_filter(metadata)

		assert result["id"] == "12345678-1234-5678-1234-567812345678"
		assert result["name"] == "test"

	def test_prepare_metadata_filter_none_values(self):
		"""Test preparing metadata filter with None values."""
		metadata = {
			"key1": "value1",
			"key2": None,
			"key3": "value3",
		}
		result = prepare_metadata_filter(metadata)

		# None values should be excluded
		assert "key1" in result
		assert "key2" not in result
		assert "key3" in result

	def test_prepare_metadata_filter_empty_dict(self):
		"""Test preparing empty metadata filter."""
		metadata = {}
		result = prepare_metadata_filter(metadata)
		assert result == {}

	def test_prepare_metadata_filter_complex_nested(self):
		"""Test preparing metadata filter with complex nested structures."""
		metadata = {
			"config": {
				"settings": {
					"enabled": True,
					"options": ["opt1", "opt2"],
					"limits": {"max": 100, "min": 0},
				}
			},
			"id": uuid.UUID("abcdef12-3456-7890-abcd-ef1234567890"),
			"nullable": None,
		}
		result = prepare_metadata_filter(metadata)

		assert "config" in result
		assert result["config"]["settings"]["enabled"] is True
		assert result["config"]["settings"]["options"] == ["opt1", "opt2"]
		assert result["config"]["settings"]["limits"] == {"max": 100, "min": 0}
		assert result["id"] == "abcdef12-3456-7890-abcd-ef1234567890"
		assert "nullable" not in result

	def test_prepare_metadata_filter_unicode(self):
		"""Test preparing metadata filter with Unicode characters."""
		metadata = {
			"emoji": "🎉",
			"chinese": "你好世界",
			"arabic": "مرحبا بالعالم",
			"mixed": {"key_😀": "value_中文"},
		}
		result = prepare_metadata_filter(metadata)

		assert result["emoji"] == "🎉"
		assert result["chinese"] == "你好世界"
		assert result["arabic"] == "مرحبا بالعالم"
		assert result["mixed"] == {"key_😀": "value_中文"}

	def test_prepare_metadata_filter_special_types(self):
		"""Test preparing metadata filter with various Python types."""
		metadata = {
			"float": 3.14159,
			"negative": -42,
			"zero": 0,
			"empty_string": "",
			"false": False,
			"tuple": (1, 2, 3),  # Will be converted to list for JSON
		}
		result = prepare_metadata_filter(metadata)

		assert result["float"] == 3.14159
		assert result["negative"] == -42
		assert result["zero"] == 0
		assert result["empty_string"] == ""
		assert result["false"] is False
		assert result["tuple"] == [1, 2, 3]  # Tuples are converted to lists


class TestIntegrationScenarios:
	"""Test integration scenarios combining multiple utility functions."""

	def test_roundtrip_channel_values(self):
		"""Test encoding and transforming channel values end-to-end."""
		# Original data
		original_data = [
			("channel1", "json", b'{"key": "value"}'),
			("channel2", "bytes", b"\x00\x01\x02\x03"),
		]

		# Simulate API format (what would come from HTTP response)
		api_format = []
		for channel, type_str, blob in original_data:
			api_format.append(
				[
					channel,
					type_str,
					encode_to_base64(blob),
				]
			)

		# Transform back to internal format
		result = transform_channel_values(api_format)

		# Verify roundtrip
		assert len(result) == 2
		assert result[0] == (b"channel1", b"json", b'{"key": "value"}')
		assert result[1] == (b"channel2", b"bytes", b"\x00\x01\x02\x03")

	def test_roundtrip_pending_writes(self):
		"""Test encoding and transforming pending writes end-to-end."""
		# Original data
		original_data = [
			("task1", "channel1", "json", b'{"action": "update"}'),
			("task2", "channel2", "binary", b"\xff\xfe\xfd"),
		]

		# Simulate API format
		api_format = []
		for task_id, channel, type_str, blob in original_data:
			api_format.append(
				[
					task_id,
					channel,
					type_str,
					encode_to_base64(blob),
				]
			)

		# Transform back to internal format
		result = transform_pending_writes(api_format)

		# Verify roundtrip
		assert len(result) == 2
		assert result[0] == (b"task1", b"channel1", b"json", b'{"action": "update"}')
		assert result[1] == (b"task2", b"channel2", b"binary", b"\xff\xfe\xfd")

	def test_metadata_filter_with_uuid_roundtrip(self):
		"""Test metadata filter with UUID conversion."""
		original_uuid = uuid.uuid4()
		metadata = {
			"request_id": original_uuid,
			"nested": {
				"session_id": uuid.uuid4(),
				"user_id": "user123",
			},
		}

		# Prepare for API
		result = prepare_metadata_filter(metadata)

		# Verify top-level UUID is converted to string
		assert isinstance(result["request_id"], str)
		# Nested dict is kept as-is (UUIDs not converted in nested structures)
		assert result["nested"]["user_id"] == "user123"

		# Verify we can reconstruct the top-level UUID
		reconstructed_uuid = uuid.UUID(result["request_id"])
		assert reconstructed_uuid == original_uuid

	def test_empty_data_handling(self):
		"""Test all functions handle empty data gracefully."""
		# Empty base64
		assert encode_to_base64(b"") == ""
		assert decode_from_base64("") == b""

		# Empty lists
		assert transform_channel_values([]) is None
		assert transform_pending_writes([]) is None

		# Empty metadata
		assert prepare_metadata_filter({}) == {}

	def test_large_data_handling(self):
		"""Test utility functions with large data."""
		# Large binary blob
		large_blob = bytes(range(256)) * 1000  # ~256KB
		encoded = encode_to_base64(large_blob)
		decoded = decode_from_base64(encoded)
		assert decoded == large_blob

		# Many channel values
		many_channels = []
		for i in range(100):
			many_channels.append(
				[
					f"channel_{i}",
					"json",
					encode_to_base64(f"data_{i}".encode()),
				]
			)
		result = transform_channel_values(many_channels)
		assert len(result) == 100

		# Large metadata
		large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
		result = prepare_metadata_filter(large_metadata)
		assert len(result) == 1000
