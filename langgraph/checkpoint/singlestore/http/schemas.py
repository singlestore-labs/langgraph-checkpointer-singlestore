"""Pydantic models for SingleStore HTTP checkpoint API.

These models provide type validation and serialization for the HTTP API,
ensuring compatibility with the Go server that uses UUIDs for identifiers.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class CheckpointData(BaseModel):
	"""Internal checkpoint data structure matching Go CheckpointData.

	This represents the core checkpoint state with versioning information.
	"""

	v: int = Field(description="Checkpoint version number")
	ts: str = Field(description="Timestamp of checkpoint creation")
	id: str = Field(description="Checkpoint identifier")
	channel_values: dict[str, Any] = Field(default_factory=dict, description="Current values for each channel")
	channel_versions: dict[str, str] = Field(default_factory=dict, description="Version identifiers for each channel")
	versions_seen: dict[str, Any] | None = Field(default=None, description="Versions seen by this checkpoint")


class BlobData(BaseModel):
	"""Blob data for checkpoint channels.

	Large or complex channel values are stored separately as blobs.
	"""

	channel: str = Field(description="Channel name")
	version: str = Field(description="Version identifier for this blob")
	type: str = Field(description="Serialization type")
	blob: str | None = Field(default=None, description="Base64 encoded binary data")


class WriteData(BaseModel):
	"""Individual write operation data.

	Represents a single write to a channel within a checkpoint.
	"""

	idx: int = Field(description="Write index/order")
	channel: str = Field(description="Target channel name")
	type: str = Field(description="Serialization type")
	blob: str = Field(description="Base64 encoded binary data")


class CheckpointRequest(BaseModel):
	"""Request for creating or updating a checkpoint.

	The Go server expects UUIDs for thread_id and checkpoint_id,
	which should be provided as strings in UUID format.
	"""

	thread_id: str = Field(description="Thread UUID as string")
	checkpoint_ns: str = Field(default="", description="Checkpoint namespace")
	checkpoint_id: str = Field(description="Checkpoint UUID as string")
	parent_checkpoint_id: str | None = Field(default=None, description="Parent checkpoint UUID as string")
	checkpoint: CheckpointData = Field(description="Checkpoint data")
	metadata: dict[str, Any] = Field(default_factory=dict, description="User-defined metadata")
	blob_data: list[BlobData] | None = Field(default=None, description="Large channel values stored separately")


class CheckpointWriteRequest(BaseModel):
	"""Request for storing checkpoint writes.

	Intermediate writes are linked to a specific checkpoint.
	"""

	thread_id: str = Field(description="Thread UUID as string")
	checkpoint_ns: str = Field(default="", description="Checkpoint namespace")
	checkpoint_id: str = Field(description="Checkpoint UUID as string")
	task_id: str = Field(description="Task identifier")
	task_path: str = Field(default="", description="Task path in execution graph")
	writes: list[WriteData] = Field(description="List of write operations")


class CheckpointResponse(BaseModel):
	"""Response containing checkpoint data.

	The Go server returns UUIDs as strings, and JSON data as raw messages.
	"""

	thread_id: str = Field(description="Thread UUID as string")
	checkpoint_ns: str = Field(description="Checkpoint namespace")
	checkpoint_id: str = Field(description="Checkpoint UUID as string")
	parent_checkpoint_id: str | None = Field(default=None, description="Parent checkpoint UUID as string")
	checkpoint: dict[str, Any] = Field(description="Checkpoint data as JSON")
	metadata: dict[str, Any] = Field(description="Metadata as JSON")
	channel_values: list[list[str]] | None = Field(
		default=None, description="Array of [channel, type, base64_blob] tuples"
	)
	pending_writes: list[list[str]] | None = Field(
		default=None, description="Array of [task_id, channel, type, base64_blob] tuples"
	)


class CheckpointListResponse(BaseModel):
	"""Response for listing checkpoints.

	Contains an array of checkpoints matching the query criteria.
	"""

	checkpoints: list[CheckpointResponse] = Field(description="List of checkpoint responses")


class CheckpointListRequest(BaseModel):
	"""Query parameters for listing checkpoints.

	All parameters are optional and used for filtering results.
	"""

	thread_id: str | None = Field(default=None, description="Filter by thread UUID")
	checkpoint_ns: str | None = Field(default=None, description="Filter by namespace")
	checkpoint_id: str | None = Field(default=None, description="Filter by checkpoint UUID")
	metadata_filter: dict[str, Any] | None = Field(default=None, description="Filter by metadata fields")
	before_checkpoint_id: str | None = Field(default=None, description="Return checkpoints before this UUID")
	limit: int | None = Field(default=None, ge=1, description="Maximum number of results")


class SetupResponse(BaseModel):
	"""Response for setup operation.

	Indicates whether database setup was successful.
	"""

	success: bool = Field(description="Whether setup succeeded")
	version: int = Field(description="Current schema version")
	message: str = Field(description="Setup status message")


class ThreadDeleteResponse(BaseModel):
	"""Response for thread deletion.

	Simplified response indicating deletion success.
	"""

	success: bool = Field(description="Whether deletion succeeded")


class ErrorDetail(BaseModel):
	"""Error detail structure.

	Consistent error format across all API endpoints.
	"""

	code: str = Field(description="Error code")
	message: str = Field(description="Human-readable error message")
	details: dict[str, Any] | None = Field(default=None, description="Additional error context")


class ErrorResponse(BaseModel):
	"""Error response structure.

	Consistent error format across all API endpoints.
	"""

	error: ErrorDetail = Field(description="Error detail")


# Helper functions for UUID conversion
def ensure_uuid_string(value: str | uuid.UUID | None) -> str | None:
	"""Convert a value to UUID string format.

	Args:
	    value: String, UUID object, or None

	Returns:
	    UUID as string or None

	Raises:
	    ValueError: If value is not a valid UUID format
	"""
	if value is None:
		return None

	if isinstance(value, uuid.UUID):
		return str(value)

	# Validate and normalize string format
	try:
		return str(uuid.UUID(value))
	except (ValueError, AttributeError) as e:
		raise ValueError(f"Invalid UUID format: {value}") from e


def generate_uuid_string() -> str:
	"""Generate a new UUID as string.

	Returns:
	    New UUID in string format
	"""
	return str(uuid.uuid4())
