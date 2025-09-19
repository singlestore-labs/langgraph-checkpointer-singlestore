"""Utility functions for type conversions and data transformations.

Handles conversions between Python and Go types, particularly for
binary data encoding.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Callable
from typing import Any

TokenGetter = Callable[[], str | None] | str | None


def encode_to_base64(data: bytes) -> str:
	"""Encode binary data to base64 string.

	Args:
	    data: Binary data to encode

	Returns:
	    Base64 encoded string
	"""
	return base64.b64encode(data).decode("utf-8")


def decode_from_base64(data: str) -> bytes:
	"""Decode base64 string to binary data.

	Args:
	    data: Base64 encoded string

	Returns:
	    Decoded binary data
	"""
	return base64.b64decode(data)


def transform_channel_values(channel_values: list[list[str]] | None) -> list[tuple[bytes, bytes, bytes]] | None:
	"""Transform channel values from API response to internal format.

	Converts from [channel, type, base64_blob] to (channel_bytes, type_bytes, blob_bytes).

	Args:
	    channel_values: List of channel value arrays from API

	Returns:
	    List of tuples with bytes data or None
	"""
	if not channel_values:
		return None

	result = []
	for item in channel_values:
		if len(item) >= 3:
			channel = item[0].encode("utf-8")
			type_str = item[1].encode("utf-8")
			blob = decode_from_base64(item[2]) if item[2] else b""
			result.append((channel, type_str, blob))

	return result


def transform_pending_writes(pending_writes: list[list[str]] | None) -> list[tuple[bytes, bytes, bytes, bytes]] | None:
	"""Transform pending writes from API response to internal format.

	Converts from [task_id, channel, type, base64_blob] to
	(task_id_bytes, channel_bytes, type_bytes, blob_bytes).

	Args:
	    pending_writes: List of pending write arrays from API

	Returns:
	    List of tuples with bytes data or None
	"""
	if not pending_writes:
		return None

	result = []
	for item in pending_writes:
		if len(item) >= 4:
			task_id = item[0].encode("utf-8")
			channel = item[1].encode("utf-8")
			type_str = item[2].encode("utf-8")
			blob = decode_from_base64(item[3]) if item[3] else b""
			result.append((task_id, channel, type_str, blob))

	return result


def prepare_metadata_filter(metadata: dict[str, Any]) -> dict[str, Any]:
	"""Prepare metadata filter for API query.

	Converts values to JSON-serializable format while maintaining the dict structure.
	Handles nested objects, UUIDs, bytes, and other non-JSON types correctly.

	Args:
	    metadata: Metadata dictionary to filter by

	Returns:
	    Prepared metadata filter dictionary with JSON-serializable values
	"""

	def convert_value(value: Any) -> Any:
		"""Recursively convert values to JSON-serializable format."""
		if isinstance(value, uuid.UUID):
			# Convert UUIDs to strings
			return str(value)
		elif isinstance(value, dict):
			# Recursively handle nested dictionaries
			return {k: convert_value(v) for k, v in value.items()}
		elif isinstance(value, list):
			# Recursively handle lists
			return [convert_value(item) for item in value]
		elif isinstance(value, tuple):
			# Convert tuples to lists for JSON compatibility
			return [convert_value(item) for item in value]
		elif isinstance(value, bytes):
			# Convert bytes to base64 string for JSON compatibility
			import base64

			return base64.b64encode(value).decode("utf-8")
		else:
			# Keep primitives (str, int, float, bool, None) as-is
			return value

	# Convert all values to JSON-serializable format
	return {key: convert_value(value) for key, value in metadata.items() if value is not None}
