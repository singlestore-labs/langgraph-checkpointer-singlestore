"""Tests for HTTP Pydantic schemas."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from langgraph.checkpoint.singlestore.http.schemas import (
	BlobData,
	CheckpointData,
	CheckpointListRequest,
	CheckpointListResponse,
	CheckpointRequest,
	CheckpointResponse,
	CheckpointWriteRequest,
	ErrorResponse,
	SetupResponse,
	ThreadDeleteResponse,
	WriteData,
)


class TestCheckpointData:
	"""Test CheckpointData model validation."""

	def test_valid_checkpoint_data(self):
		"""Test creating valid CheckpointData."""
		data = {
			"v": 1,
			"ts": "2024-01-01T00:00:00",
			"id": "checkpoint-1",
			"channel_values": {"channel1": "value1"},
			"channel_versions": {"channel1": "version1"},
			"versions_seen": {"source": {"channel1": "version1"}},
		}
		checkpoint = CheckpointData(**data)
		assert checkpoint.v == 1
		assert checkpoint.ts == "2024-01-01T00:00:00"
		assert checkpoint.id == "checkpoint-1"
		assert checkpoint.channel_values == {"channel1": "value1"}

	def test_checkpoint_data_optional_fields(self):
		"""Test CheckpointData with optional fields omitted."""
		data = {
			"v": 1,
			"ts": "2024-01-01T00:00:00",
			"id": "checkpoint-1",
		}
		checkpoint = CheckpointData(**data)
		assert checkpoint.channel_values == {}
		assert checkpoint.channel_versions == {}
		assert checkpoint.versions_seen is None

	def test_checkpoint_data_missing_required(self):
		"""Test CheckpointData validation with missing required fields."""
		with pytest.raises(ValidationError) as exc_info:
			CheckpointData(v=1, ts="2024-01-01T00:00:00")
		errors = exc_info.value.errors()
		assert any(error["loc"] == ("id",) for error in errors)

	def test_checkpoint_data_invalid_types(self):
		"""Test CheckpointData validation with invalid types."""
		with pytest.raises(ValidationError) as exc_info:
			CheckpointData(
				v="not_an_int",  # Should be int
				ts="2024-01-01T00:00:00",
				id="checkpoint-1",
			)
		errors = exc_info.value.errors()
		assert any("v" in str(error["loc"]) for error in errors)


class TestBlobData:
	"""Test BlobData model validation."""

	def test_valid_blob_data(self):
		"""Test creating valid BlobData."""
		data = {
			"channel": "channel1",
			"version": "version1",
			"type": "json",
			"blob": "eyJrZXkiOiAidmFsdWUifQ==",  # base64 encoded {"key": "value"}
		}
		blob = BlobData(**data)
		assert blob.channel == "channel1"
		assert blob.version == "version1"
		assert blob.type == "json"
		assert blob.blob == "eyJrZXkiOiAidmFsdWUifQ=="

	def test_blob_data_null_blob(self):
		"""Test BlobData with null blob."""
		data = {
			"channel": "channel1",
			"version": "version1",
			"type": "json",
			"blob": None,
		}
		blob = BlobData(**data)
		assert blob.blob is None

	def test_blob_data_missing_required(self):
		"""Test BlobData validation with missing required fields."""
		with pytest.raises(ValidationError) as exc_info:
			BlobData(channel="channel1", version="version1")
		errors = exc_info.value.errors()
		assert any(error["loc"] == ("type",) for error in errors)


class TestWriteData:
	"""Test WriteData model validation."""

	def test_valid_write_data(self):
		"""Test creating valid WriteData."""
		data = {
			"idx": 0,
			"channel": "channel1",
			"type": "json",
			"blob": "eyJrZXkiOiAidmFsdWUifQ==",
		}
		write = WriteData(**data)
		assert write.idx == 0
		assert write.channel == "channel1"
		assert write.type == "json"
		assert write.blob == "eyJrZXkiOiAidmFsdWUifQ=="

	def test_write_data_negative_idx(self):
		"""Test WriteData with negative index."""
		data = {
			"idx": -1,
			"channel": "channel1",
			"type": "json",
			"blob": "data",
		}
		write = WriteData(**data)
		assert write.idx == -1  # Should allow negative indices

	def test_write_data_missing_blob(self):
		"""Test WriteData validation with missing blob."""
		with pytest.raises(ValidationError) as exc_info:
			WriteData(idx=0, channel="channel1", type="json")
		errors = exc_info.value.errors()
		assert any(error["loc"] == ("blob",) for error in errors)


class TestCheckpointRequest:
	"""Test CheckpointRequest model validation."""

	def test_valid_checkpoint_request(self):
		"""Test creating valid CheckpointRequest."""
		checkpoint_data = CheckpointData(
			v=1,
			ts="2024-01-01T00:00:00",
			id="checkpoint-1",
		)
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"checkpoint": checkpoint_data,
			"metadata": {"source": "test"},
		}
		request = CheckpointRequest(**data)
		assert request.thread_id == "thread-1"
		assert request.checkpoint_ns == ""
		assert request.checkpoint_id == "checkpoint-1"
		assert request.metadata == {"source": "test"}

	def test_checkpoint_request_with_blobs(self):
		"""Test CheckpointRequest with blob data."""
		checkpoint_data = CheckpointData(
			v=1,
			ts="2024-01-01T00:00:00",
			id="checkpoint-1",
		)
		blob_data = BlobData(
			channel="channel1",
			version="version1",
			type="json",
			blob="data",
		)
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"checkpoint": checkpoint_data,
			"metadata": {},
			"blob_data": [blob_data],
		}
		request = CheckpointRequest(**data)
		assert len(request.blob_data) == 1
		assert request.blob_data[0].channel == "channel1"

	def test_checkpoint_request_with_parent(self):
		"""Test CheckpointRequest with parent checkpoint ID."""
		checkpoint_data = CheckpointData(
			v=1,
			ts="2024-01-01T00:00:00",
			id="checkpoint-2",
		)
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-2",
			"parent_checkpoint_id": "checkpoint-1",
			"checkpoint": checkpoint_data,
			"metadata": {},
		}
		request = CheckpointRequest(**data)
		assert request.parent_checkpoint_id == "checkpoint-1"

	def test_checkpoint_request_missing_checkpoint(self):
		"""Test CheckpointRequest validation with missing checkpoint."""
		with pytest.raises(ValidationError) as exc_info:
			CheckpointRequest(
				thread_id="thread-1",
				checkpoint_ns="",
				checkpoint_id="checkpoint-1",
				metadata={},
			)
		errors = exc_info.value.errors()
		assert any(error["loc"] == ("checkpoint",) for error in errors)


class TestCheckpointResponse:
	"""Test CheckpointResponse model validation."""

	def test_valid_checkpoint_response(self):
		"""Test creating valid CheckpointResponse."""
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"checkpoint": {"v": 1, "ts": "2024-01-01T00:00:00", "id": "checkpoint-1"},
			"metadata": {"source": "test"},
		}
		response = CheckpointResponse(**data)
		assert response.thread_id == "thread-1"
		assert response.checkpoint_ns == ""
		assert response.checkpoint_id == "checkpoint-1"
		assert response.metadata == {"source": "test"}

	def test_checkpoint_response_with_values(self):
		"""Test CheckpointResponse with channel values and pending writes."""
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"checkpoint": {"v": 1, "ts": "2024-01-01T00:00:00", "id": "checkpoint-1"},
			"metadata": {},
			"channel_values": [["channel1", "json", "data"]],
			"pending_writes": [["task1", "channel1", "json", "data"]],
		}
		response = CheckpointResponse(**data)
		assert len(response.channel_values) == 1
		assert response.channel_values[0] == ["channel1", "json", "data"]
		assert len(response.pending_writes) == 1
		assert response.pending_writes[0] == ["task1", "channel1", "json", "data"]

	def test_checkpoint_response_with_parent(self):
		"""Test CheckpointResponse with parent checkpoint."""
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-2",
			"parent_checkpoint_id": "checkpoint-1",
			"checkpoint": {"v": 1, "ts": "2024-01-01T00:00:00", "id": "checkpoint-2"},
			"metadata": {},
		}
		response = CheckpointResponse(**data)
		assert response.parent_checkpoint_id == "checkpoint-1"


class TestCheckpointListRequest:
	"""Test CheckpointListRequest model validation."""

	def test_valid_list_request(self):
		"""Test creating valid CheckpointListRequest."""
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"limit": 10,
		}
		request = CheckpointListRequest(**data)
		assert request.thread_id == "thread-1"
		assert request.checkpoint_ns == ""
		assert request.limit == 10

	def test_list_request_with_filter(self):
		"""Test CheckpointListRequest with metadata filter."""
		data = {
			"thread_id": "thread-1",
			"metadata_filter": {"source": "test", "step": 5},
		}
		request = CheckpointListRequest(**data)
		assert request.metadata_filter == {"source": "test", "step": 5}

	def test_list_request_with_before(self):
		"""Test CheckpointListRequest with before checkpoint."""
		data = {
			"thread_id": "thread-1",
			"before_checkpoint_id": "checkpoint-5",
			"limit": 5,
		}
		request = CheckpointListRequest(**data)
		assert request.before_checkpoint_id == "checkpoint-5"
		assert request.limit == 5

	def test_list_request_all_optional(self):
		"""Test CheckpointListRequest with all fields optional."""
		request = CheckpointListRequest()
		assert request.thread_id is None
		assert request.checkpoint_ns is None
		assert request.checkpoint_id is None
		assert request.metadata_filter is None
		assert request.before_checkpoint_id is None
		assert request.limit is None


class TestCheckpointListResponse:
	"""Test CheckpointListResponse model validation."""

	def test_valid_list_response(self):
		"""Test creating valid CheckpointListResponse."""
		checkpoint_response = CheckpointResponse(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint={"v": 1, "ts": "2024-01-01T00:00:00", "id": "checkpoint-1"},
			metadata={},
		)
		data = {
			"checkpoints": [checkpoint_response],
		}
		response = CheckpointListResponse(**data)
		assert len(response.checkpoints) == 1

	def test_list_response_empty(self):
		"""Test CheckpointListResponse with empty list."""
		data = {
			"checkpoints": [],
		}
		response = CheckpointListResponse(**data)
		assert len(response.checkpoints) == 0

	def test_list_response_missing_checkpoints(self):
		"""Test CheckpointListResponse requires checkpoints field."""
		with pytest.raises(ValidationError) as exc_info:
			CheckpointListResponse()

		errors = exc_info.value.errors()
		assert len(errors) == 1
		assert errors[0]["loc"] == ("checkpoints",)
		assert errors[0]["type"] == "missing"


class TestCheckpointWriteRequest:
	"""Test CheckpointWriteRequest model validation."""

	def test_valid_write_request(self):
		"""Test creating valid CheckpointWriteRequest."""
		write_data = WriteData(
			idx=0,
			channel="channel1",
			type="json",
			blob="data",
		)
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"task_id": "task-1",
			"writes": [write_data],
		}
		request = CheckpointWriteRequest(**data)
		assert request.thread_id == "thread-1"
		assert request.task_id == "task-1"
		assert len(request.writes) == 1

	def test_write_request_with_task_path(self):
		"""Test CheckpointWriteRequest with task path."""
		write_data = WriteData(
			idx=0,
			channel="channel1",
			type="json",
			blob="data",
		)
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"task_id": "task-1",
			"task_path": "/path/to/task",
			"writes": [write_data],
		}
		request = CheckpointWriteRequest(**data)
		assert request.task_path == "/path/to/task"

	def test_write_request_multiple_writes(self):
		"""Test CheckpointWriteRequest with multiple writes."""
		writes = [WriteData(idx=i, channel=f"channel{i}", type="json", blob=f"data{i}") for i in range(5)]
		data = {
			"thread_id": "thread-1",
			"checkpoint_ns": "",
			"checkpoint_id": "checkpoint-1",
			"task_id": "task-1",
			"writes": writes,
		}
		request = CheckpointWriteRequest(**data)
		assert len(request.writes) == 5
		assert request.writes[2].channel == "channel2"


class TestErrorResponse:
	"""Test ErrorResponse model validation."""

	def test_valid_error_response(self):
		"""Test creating valid ErrorResponse."""
		data = {
			"error": {
				"code": "ValidationError",
				"message": "Invalid checkpoint ID",
			}
		}
		response = ErrorResponse(**data)
		assert response.error.code == "ValidationError"
		assert response.error.message == "Invalid checkpoint ID"
		assert response.error.details is None

	def test_error_response_with_details(self):
		"""Test ErrorResponse with additional details."""
		data = {
			"error": {
				"code": "DatabaseError",
				"message": "Connection timeout",
				"details": {"timeout": 30, "host": "localhost"},
			}
		}
		response = ErrorResponse(**data)
		assert response.error.code == "DatabaseError"
		assert response.error.message == "Connection timeout"
		assert response.error.details == {"timeout": 30, "host": "localhost"}

	def test_error_response_minimal(self):
		"""Test ErrorResponse with minimal fields."""
		data = {
			"error": {
				"code": "UnknownError",
				"message": "Something went wrong",
			}
		}
		response = ErrorResponse(**data)
		assert response.error.code == "UnknownError"
		assert response.error.message == "Something went wrong"
		assert response.error.details is None


class TestSetupResponse:
	"""Test SetupResponse model validation."""

	def test_valid_setup_response_success(self):
		"""Test creating valid successful SetupResponse."""
		data = {
			"success": True,
			"message": "Database setup completed",
			"version": 1,  # version is an integer
		}
		response = SetupResponse(**data)
		assert response.success is True
		assert response.message == "Database setup completed"
		assert response.version == 1

	def test_valid_setup_response_failure(self):
		"""Test creating valid failed SetupResponse."""
		data = {
			"success": False,
			"message": "Failed to create tables",
			"version": 0,  # version is an integer
		}
		response = SetupResponse(**data)
		assert response.success is False
		assert response.message == "Failed to create tables"
		assert response.version == 0

	def test_setup_response_missing_message(self):
		"""Test SetupResponse validation with missing message."""
		with pytest.raises(ValidationError) as exc_info:
			SetupResponse(success=True, version=1)
		errors = exc_info.value.errors()
		assert any(error["loc"] == ("message",) for error in errors)


class TestThreadDeleteResponse:
	"""Test ThreadDeleteResponse model validation."""

	def test_valid_delete_response_success(self):
		"""Test creating valid successful ThreadDeleteResponse."""
		data = {
			"success": True,
		}
		response = ThreadDeleteResponse(**data)
		assert response.success is True

	def test_valid_delete_response_failure(self):
		"""Test creating valid failed ThreadDeleteResponse."""
		data = {
			"success": False,
		}
		response = ThreadDeleteResponse(**data)
		assert response.success is False

	def test_delete_response_missing_success(self):
		"""Test ThreadDeleteResponse validation with missing success."""
		with pytest.raises(ValidationError) as exc_info:
			ThreadDeleteResponse()
		errors = exc_info.value.errors()
		assert any(error["loc"] == ("success",) for error in errors)


class TestModelSerialization:
	"""Test model serialization and deserialization."""

	def test_checkpoint_request_json_serialization(self):
		"""Test CheckpointRequest JSON serialization."""
		checkpoint_data = CheckpointData(
			v=1,
			ts="2024-01-01T00:00:00",
			id="checkpoint-1",
		)
		request = CheckpointRequest(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint=checkpoint_data,
			metadata={"source": "test"},
		)

		# Serialize to JSON
		json_str = request.model_dump_json()
		data = json.loads(json_str)

		# Deserialize back
		request2 = CheckpointRequest(**data)
		assert request2.thread_id == request.thread_id
		assert request2.checkpoint_id == request.checkpoint_id
		assert request2.metadata == request.metadata

	def test_model_exclude_none(self):
		"""Test model serialization with exclude_none."""
		request = CheckpointListRequest(
			thread_id="thread-1",
			limit=10,
		)

		# Serialize with exclude_none
		data = request.model_dump(exclude_none=True)

		# Should not include None fields
		assert "checkpoint_ns" not in data
		assert "checkpoint_id" not in data
		assert "metadata_filter" not in data
		assert "before_checkpoint_id" not in data

		# Should include non-None fields
		assert data["thread_id"] == "thread-1"
		assert data["limit"] == 10

	def test_nested_model_validation(self):
		"""Test validation of nested models."""
		# Invalid nested CheckpointData
		with pytest.raises(ValidationError) as exc_info:
			CheckpointRequest(
				thread_id="thread-1",
				checkpoint_ns="",
				checkpoint_id="checkpoint-1",
				checkpoint={"v": "not_an_int", "ts": "2024-01-01T00:00:00"},  # Invalid v type
				metadata={},
			)
		errors = exc_info.value.errors()
		# Should have validation error for nested field
		assert any("checkpoint" in str(error["loc"]) for error in errors)


class TestEdgeCases:
	"""Test edge cases and special scenarios."""

	def test_empty_strings(self):
		"""Test models with empty string values."""
		request = CheckpointRequest(
			thread_id="",  # Empty thread_id
			checkpoint_ns="",
			checkpoint_id="",  # Empty checkpoint_id
			checkpoint=CheckpointData(v=1, ts="", id=""),  # Empty timestamps and IDs
			metadata={},
		)
		assert request.thread_id == ""
		assert request.checkpoint_id == ""

	def test_large_metadata(self):
		"""Test models with large metadata objects."""
		large_metadata = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
		request = CheckpointRequest(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint=CheckpointData(v=1, ts="2024-01-01T00:00:00", id="checkpoint-1"),
			metadata=large_metadata,
		)
		assert len(request.metadata) == 100

	def test_unicode_in_fields(self):
		"""Test models with Unicode characters."""
		request = CheckpointRequest(
			thread_id="thread-😀",
			checkpoint_ns="namespace-中文",
			checkpoint_id="checkpoint-1",
			checkpoint=CheckpointData(v=1, ts="2024-01-01T00:00:00", id="checkpoint-1"),
			metadata={"emoji": "🎉", "chinese": "你好", "arabic": "مرحبا"},
		)
		assert request.thread_id == "thread-😀"
		assert request.checkpoint_ns == "namespace-中文"
		assert request.metadata["emoji"] == "🎉"

	def test_special_characters_in_json(self):
		"""Test models with special JSON characters."""
		metadata = {
			"quotes": 'He said "Hello"',
			"backslash": "path\\to\\file",
			"newline": "line1\nline2",
			"tab": "col1\tcol2",
		}
		request = CheckpointRequest(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint=CheckpointData(v=1, ts="2024-01-01T00:00:00", id="checkpoint-1"),
			metadata=metadata,
		)

		# Should serialize and deserialize correctly
		json_str = request.model_dump_json()
		data = json.loads(json_str)
		request2 = CheckpointRequest(**data)
		assert request2.metadata["quotes"] == 'He said "Hello"'
		assert request2.metadata["backslash"] == "path\\to\\file"


class TestUUIDValidation:
	"""Test UUID validation functions."""

	def test_ensure_uuid_string_with_valid_string(self):
		"""Test ensure_uuid_string with valid UUID string."""
		from langgraph.checkpoint.singlestore.http.schemas import ensure_uuid_string

		uuid_str = "12345678-1234-5678-1234-567812345678"
		result = ensure_uuid_string(uuid_str)
		assert result == uuid_str

	def test_ensure_uuid_string_with_uuid_object(self):
		"""Test ensure_uuid_string with UUID object."""
		import uuid
		from langgraph.checkpoint.singlestore.http.schemas import ensure_uuid_string

		uuid_obj = uuid.UUID("12345678-1234-5678-1234-567812345678")
		result = ensure_uuid_string(uuid_obj)
		assert result == "12345678-1234-5678-1234-567812345678"

	def test_ensure_uuid_string_with_none(self):
		"""Test ensure_uuid_string with None value."""
		from langgraph.checkpoint.singlestore.http.schemas import ensure_uuid_string

		result = ensure_uuid_string(None)
		assert result is None

	def test_ensure_uuid_string_with_invalid_string(self):
		"""Test ensure_uuid_string with invalid UUID string."""
		from langgraph.checkpoint.singlestore.http.schemas import ensure_uuid_string

		with pytest.raises(ValueError) as exc_info:
			ensure_uuid_string("not-a-uuid")
		assert "Invalid UUID format" in str(exc_info.value)

	def test_ensure_uuid_string_with_attribute_error(self):
		"""Test ensure_uuid_string with object that causes AttributeError."""
		from langgraph.checkpoint.singlestore.http.schemas import ensure_uuid_string

		with pytest.raises(ValueError) as exc_info:
			ensure_uuid_string(12345)  # Invalid type
		assert "Invalid UUID format" in str(exc_info.value)

	def test_ensure_uuid_string_with_malformed_uuid(self):
		"""Test ensure_uuid_string with malformed UUID strings."""
		from langgraph.checkpoint.singlestore.http.schemas import ensure_uuid_string

		invalid_uuids = [
			"12345678",  # Too short
			"12345678-1234-5678-1234-567812345678-extra",  # Too long
			"XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",  # Invalid characters
			"12345678-1234-5678-1234",  # Missing part
		]

		for invalid_uuid in invalid_uuids:
			with pytest.raises(ValueError) as exc_info:
				ensure_uuid_string(invalid_uuid)
			assert "Invalid UUID format" in str(exc_info.value)

	def test_generate_uuid_string(self):
		"""Test generate_uuid_string function."""
		import uuid
		from langgraph.checkpoint.singlestore.http.schemas import generate_uuid_string

		uuid_str = generate_uuid_string()

		# Should be a valid UUID string
		assert isinstance(uuid_str, str)

		# Should be parseable as UUID
		try:
			uuid_obj = uuid.UUID(uuid_str)
			assert str(uuid_obj) == uuid_str
		except ValueError:
			pytest.fail(f"Generated UUID string is not valid: {uuid_str}")

		# Should generate unique UUIDs
		uuid_str2 = generate_uuid_string()
		assert uuid_str != uuid_str2


class TestSchemaContracts:
	"""Test that schemas maintain their contracts and field requirements."""

	def test_checkpoint_list_request_required_fields(self):
		"""Test that CheckpointListRequest has no required fields."""
		# All fields should be optional for flexible querying
		request = CheckpointListRequest()
		assert request.thread_id is None
		assert request.checkpoint_ns is None
		assert request.checkpoint_id is None
		assert request.metadata_filter is None
		assert request.before_checkpoint_id is None
		assert request.limit is None

	def test_checkpoint_list_request_limit_validation(self):
		"""Test that limit field has proper validation."""
		# Limit must be >= 1
		with pytest.raises(ValidationError) as exc_info:
			CheckpointListRequest(limit=0)
		errors = exc_info.value.errors()
		assert any("greater_than_equal" in str(error) for error in errors)

		with pytest.raises(ValidationError) as exc_info:
			CheckpointListRequest(limit=-1)
		errors = exc_info.value.errors()
		assert any("greater_than_equal" in str(error) for error in errors)

	def test_checkpoint_response_required_fields(self):
		"""Test that CheckpointResponse maintains its required fields."""
		# These fields must always be required
		with pytest.raises(ValidationError) as exc_info:
			CheckpointResponse()
		errors = exc_info.value.errors()
		required_fields = {"thread_id", "checkpoint_ns", "checkpoint_id", "checkpoint", "metadata"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields == missing_fields

	def test_checkpoint_data_required_fields(self):
		"""Test that CheckpointData maintains its required fields."""
		# v, ts, and id are required
		with pytest.raises(ValidationError) as exc_info:
			CheckpointData()
		errors = exc_info.value.errors()
		required_fields = {"v", "ts", "id"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields == missing_fields

	def test_checkpoint_data_type_validation(self):
		"""Test that CheckpointData validates field types correctly."""
		# v must be int
		with pytest.raises(ValidationError) as exc_info:
			CheckpointData(v="string", ts="2024-01-01", id="test")
		errors = exc_info.value.errors()
		assert any("int" in str(error) for error in errors)

	def test_blob_data_required_fields(self):
		"""Test that BlobData maintains its required fields."""
		# channel, version, and type are required
		with pytest.raises(ValidationError) as exc_info:
			BlobData()
		errors = exc_info.value.errors()
		required_fields = {"channel", "version", "type"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields == missing_fields

	def test_checkpoint_request_required_fields(self):
		"""Test that CheckpointRequest maintains its required fields."""
		# These fields must always be required for PUT operations
		with pytest.raises(ValidationError) as exc_info:
			CheckpointRequest()
		errors = exc_info.value.errors()
		required_fields = {"thread_id", "checkpoint_id", "checkpoint"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		# checkpoint_ns has a default, so it's not in missing
		assert required_fields.issubset(missing_fields)

	def test_checkpoint_write_request_required_fields(self):
		"""Test that CheckpointWriteRequest maintains its required fields."""
		# These fields must always be required for write operations
		with pytest.raises(ValidationError) as exc_info:
			CheckpointWriteRequest()
		errors = exc_info.value.errors()
		required_fields = {"thread_id", "checkpoint_id", "task_id", "writes"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		# checkpoint_ns and task_path have defaults
		assert required_fields.issubset(missing_fields)

	def test_setup_response_required_fields(self):
		"""Test that SetupResponse maintains its required fields."""
		# success, version, and message are required
		with pytest.raises(ValidationError) as exc_info:
			SetupResponse()
		errors = exc_info.value.errors()
		required_fields = {"success", "version", "message"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields == missing_fields

	def test_error_response_required_fields(self):
		"""Test that ErrorResponse maintains its required fields."""
		# Only error field is required (which contains ErrorDetail)
		with pytest.raises(ValidationError) as exc_info:
			ErrorResponse()
		errors = exc_info.value.errors()
		required_fields = {"error"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields == missing_fields

		# Test that ErrorDetail requires code and message
		with pytest.raises(ValidationError) as exc_info:
			ErrorResponse(error={})
		errors = exc_info.value.errors()
		# Check that code and message are required in the nested error field
		assert any(error["loc"] == ("error", "code") and error["type"] == "missing" for error in errors)
		assert any(error["loc"] == ("error", "message") and error["type"] == "missing" for error in errors)

	def test_model_dump_exclude_none_behavior(self):
		"""Test that model_dump(exclude_none=True) works as expected."""
		# Create request with only some fields
		request = CheckpointListRequest(thread_id="test-thread", limit=10)
		dumped = request.model_dump(exclude_none=True)

		# Should only include non-None fields
		assert dumped == {"thread_id": "test-thread", "limit": 10}
		assert "checkpoint_ns" not in dumped
		assert "metadata_filter" not in dumped

		# Create request with all None fields
		empty_request = CheckpointListRequest()
		empty_dumped = empty_request.model_dump(exclude_none=True)
		assert empty_dumped == {}

	def test_checkpoint_response_channel_values_format(self):
		"""Test that channel_values field accepts expected format."""
		# channel_values should be list of lists of strings
		response = CheckpointResponse(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint={"v": 1},
			metadata={},
			channel_values=[["channel1", "type1", "base64data"]],
		)
		assert response.channel_values == [["channel1", "type1", "base64data"]]

		# Should also accept None
		response2 = CheckpointResponse(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint={"v": 1},
			metadata={},
			channel_values=None,
		)
		assert response2.channel_values is None

	def test_checkpoint_response_pending_writes_format(self):
		"""Test that pending_writes field accepts expected format."""
		# pending_writes should be list of lists of strings
		response = CheckpointResponse(
			thread_id="thread-1",
			checkpoint_ns="",
			checkpoint_id="checkpoint-1",
			checkpoint={"v": 1},
			metadata={},
			pending_writes=[["task1", "channel1", "type1", "base64data"]],
		)
		assert response.pending_writes == [["task1", "channel1", "type1", "base64data"]]
