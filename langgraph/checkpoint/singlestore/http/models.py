"""HTTP request/response models for SingleStore checkpointer HTTP API."""

from __future__ import annotations

from typing import Any, TypedDict


class CheckpointRequest(TypedDict):
	"""Request body for creating/updating a checkpoint."""

	thread_id: str
	checkpoint_ns: str
	checkpoint_id: str
	parent_checkpoint_id: str | None
	checkpoint: dict[str, Any]
	metadata: dict[str, Any]
	blob_data: list[BlobData] | None


class BlobData(TypedDict):
	"""Blob data for checkpoint channels."""

	channel: str
	version: str
	type: str
	blob: str | None  # Base64 encoded binary data


class CheckpointWriteRequest(TypedDict):
	"""Request body for checkpoint writes."""

	thread_id: str
	checkpoint_ns: str
	checkpoint_id: str
	task_id: str
	task_path: str
	writes: list[WriteData]


class WriteData(TypedDict):
	"""Individual write data."""

	idx: int
	channel: str
	type: str
	blob: str  # Base64 encoded binary data


class CheckpointListRequest(TypedDict, total=False):
	"""Query parameters for listing checkpoints."""

	thread_id: str | None
	checkpoint_ns: str | None
	checkpoint_id: str | None
	metadata_filter: dict[str, Any] | None
	before_checkpoint_id: str | None
	limit: int | None


class CheckpointResponse(TypedDict):
	"""Response for a single checkpoint."""

	thread_id: str
	checkpoint_ns: str
	checkpoint_id: str
	parent_checkpoint_id: str | None
	checkpoint: dict[str, Any]
	metadata: dict[str, Any]
	channel_values: list[list[Any]] | None  # Encoded channel values
	pending_writes: list[list[Any]] | None  # Encoded pending writes


class CheckpointListResponse(TypedDict):
	"""Response for listing checkpoints."""

	checkpoints: list[CheckpointResponse]
	total: int | None


class SetupResponse(TypedDict):
	"""Response for setup operation."""

	success: bool
	version: int
	message: str | None


class DeleteThreadResponse(TypedDict):
	"""Response for thread deletion."""

	success: bool
	deleted_checkpoints: int
	deleted_blobs: int
	deleted_writes: int


class ErrorResponse(TypedDict):
	"""Error response structure."""

	error: str
	message: str
	status_code: int
	details: dict[str, Any] | None
