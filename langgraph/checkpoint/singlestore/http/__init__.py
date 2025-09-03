"""HTTP-based SingleStore checkpointer implementation."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from langgraph.checkpoint.base import (
	WRITES_IDX_MAP,
	ChannelVersions,
	Checkpoint,
	CheckpointMetadata,
	CheckpointTuple,
	get_checkpoint_id,
	get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.singlestore.base import BaseSingleStoreSaver

from .client import HTTPClient, HTTPClientError, RetryConfig
from .constants import INVALID_REQUEST_PAYLOAD, INVALID_RESPONSE
from .schemas import (
	BlobData,
	CheckpointData,
	CheckpointListRequest,
	CheckpointListResponse,
	CheckpointRequest,
	CheckpointResponse,
	CheckpointWriteRequest,
	SetupResponse,
	ThreadDeleteResponse,
	WriteData,
)
from .utils import (
	encode_to_base64,
	prepare_metadata_filter,
	transform_channel_values,
	transform_pending_writes,
)


class HTTPSingleStoreSaver(BaseSingleStoreSaver):
	"""HTTP-based checkpointer that stores checkpoints via API calls."""

	lock: threading.Lock

	def __init__(
		self,
		base_url: str,
		base_path: str = "",
		api_key: str | None = None,
		serde: SerializerProtocol | None = None,
		timeout: float = 30.0,
		retry_config: RetryConfig | None = None,
		pool_connections: int = 10,
		pool_maxsize: int = 20,
	) -> None:
		"""Initialize HTTP-based SingleStore checkpointer.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			base_path: Base path to prepend to all endpoints
			api_key: Optional API key for authentication
			serde: Serializer for checkpoint data
			timeout: Request timeout in seconds
			retry_config: Configuration for request retries
			pool_connections: Number of connection pool connections
			pool_maxsize: Maximum size of connection pool
		"""
		super().__init__(serde=serde)
		self.base_url = base_url
		self.base_path = base_path
		self.api_key = api_key
		self.timeout = timeout
		self.retry_config = retry_config
		self.pool_connections = pool_connections
		self.pool_maxsize = pool_maxsize
		self.lock = threading.Lock()
		self._client: HTTPClient | None = None

	@contextmanager
	def _get_client(self) -> Iterator[HTTPClient]:
		"""Get or create HTTP client."""
		if self._client is None:
			client = HTTPClient(
				base_url=self.base_url,
				base_path=self.base_path,
				api_key=self.api_key,
				timeout=self.timeout,
				retry_config=self.retry_config,
				pool_connections=self.pool_connections,
				pool_maxsize=self.pool_maxsize,
			)
			with client.create() as http_client:
				yield http_client
		else:
			yield self._client

	@classmethod
	@contextmanager
	def from_url(
		cls,
		base_url: str,
		base_path: str = "",
		api_key: str | None = None,
		**kwargs,
	) -> Iterator[HTTPSingleStoreSaver]:
		"""Create a new HTTPSingleStoreSaver instance from a URL.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			base_path: Base path to prepend to all endpoints
			api_key: Optional API key for authentication
			**kwargs: Additional arguments for HTTPSingleStoreSaver

		Yields:
			HTTPSingleStoreSaver instance
		"""
		saver = cls(base_url=base_url, base_path=base_path, api_key=api_key, **kwargs)
		client = HTTPClient(
			base_url=base_url,
			base_path=base_path,
			api_key=api_key,
			timeout=kwargs.get("timeout", 30.0),
			retry_config=kwargs.get("retry_config"),
			pool_connections=kwargs.get("pool_connections", 10),
			pool_maxsize=kwargs.get("pool_maxsize", 20),
		)
		with client.create() as http_client:
			saver._client = http_client
			try:
				yield saver
			finally:
				saver._client = None

	def setup(self) -> None:
		"""Set up the checkpoint database via HTTP API."""
		with self.lock, self._get_client() as client:
			try:
				response = client.post("/checkpoints/setup", {})

				# Wrap response parsing to catch validation errors
				try:
					parsed = SetupResponse(**response)
				except ValidationError as e:
					raise HTTPClientError(
						message="Invalid setup response from server",
						error_code="INVALID_RESPONSE",
						details={
							"response": response,
							"errors": e.errors(),
						},
					) from e

				if not parsed.success:
					raise HTTPClientError(f"Setup failed: {parsed.message}")
			except HTTPClientError:
				raise

	def list(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> Iterator[CheckpointTuple]:
		"""List checkpoints from the database via HTTP API."""
		# Build request data as dict for Pydantic model
		request_dict = {}

		if config:
			request_dict["thread_id"] = config["configurable"]["thread_id"]
			checkpoint_ns = config["configurable"].get("checkpoint_ns")
			if checkpoint_ns is not None:
				request_dict["checkpoint_ns"] = checkpoint_ns
			if checkpoint_id := get_checkpoint_id(config):
				request_dict["checkpoint_id"] = checkpoint_id

		if filter:
			request_dict["metadata_filter"] = prepare_metadata_filter(filter)

		if before and (before_id := get_checkpoint_id(before)):
			request_dict["before_checkpoint_id"] = before_id

		if limit is not None:
			request_dict["limit"] = limit

		# Create Pydantic model and convert to dict for API call
		try:
			request_model = CheckpointListRequest(**request_dict)
		except ValidationError as e:
			# Validation failed - raise before any network operations
			raise HTTPClientError(
				message="Invalid request payload",
				error_code=INVALID_REQUEST_PAYLOAD,
				details={
					"request_payload": request_dict,
					"errors": e.errors(),
				},
			) from e
		request_data = request_model.model_dump(exclude_none=True)

		# JSON-serialize metadata_filter for query parameter if present
		# This ensures proper encoding of booleans (true/false) and nested structures
		if request_data.get("metadata_filter"):
			import json

			request_data["metadata_filter"] = json.dumps(request_data["metadata_filter"])

		with self._get_client() as client:
			try:
				response = client.get("/checkpoints", params=request_data)

				# Wrap response parsing to catch validation errors
				try:
					parsed = CheckpointListResponse(**response)
				except ValidationError as e:
					raise HTTPClientError(
						message="Invalid response from server",
						error_code=INVALID_RESPONSE,
						details={
							"response": response,
							"errors": e.errors(),
						},
					) from e

				checkpoints = [cp.model_dump() for cp in parsed.checkpoints]

				for checkpoint_data in checkpoints:
					yield self._checkpoint_response_to_tuple(checkpoint_data)
			except HTTPClientError:
				raise

	def get(self, config: RunnableConfig) -> Checkpoint | None:
		"""Get a checkpoint from the database via HTTP API."""
		checkpoint_tuple = self.get_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
		"""Get a checkpoint tuple from the database via HTTP API."""
		thread_id = config["configurable"]["thread_id"]
		checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
		checkpoint_id = get_checkpoint_id(config)

		with self._get_client() as client:
			try:
				if checkpoint_id:
					# Get specific checkpoint
					path = f"/checkpoints/{thread_id}/{checkpoint_id}"
				else:
					# Get latest checkpoint
					path = f"/checkpoints/{thread_id}/latest"

				# Add checkpoint_ns as query parameter if provided
				params = {}
				if checkpoint_ns:
					params["checkpoint_ns"] = checkpoint_ns

				response = client.get(path, params=params)

				# Wrap response parsing to catch validation errors
				try:
					parsed = CheckpointResponse(**response)
				except ValidationError as e:
					raise HTTPClientError(
						message="Invalid checkpoint response from server",
						error_code=INVALID_RESPONSE,
						details={
							"response": response,
							"errors": e.errors(),
						},
					) from e

				response_dict = parsed.model_dump()
				return self._checkpoint_response_to_tuple(response_dict)
			except HTTPClientError as e:
				if e.status_code == 404:
					return None
				raise

	def put(
		self,
		config: RunnableConfig,
		checkpoint: Checkpoint,
		metadata: CheckpointMetadata,
		new_versions: ChannelVersions,
	) -> RunnableConfig:
		"""Save a checkpoint to the database via HTTP API."""
		configurable = config["configurable"].copy()
		thread_id = configurable.pop("thread_id")
		checkpoint_ns = configurable.pop("checkpoint_ns")
		parent_checkpoint_id = configurable.pop("checkpoint_id", None)

		# Prepare checkpoint data
		copy = checkpoint.copy()
		copy["channel_values"] = copy["channel_values"].copy()

		# Separate blob values from inline values
		blob_values = {}
		for k, v in checkpoint["channel_values"].items():
			if v is None or isinstance(v, str | int | float | bool):
				pass
			else:
				blob_values[k] = copy["channel_values"].pop(k)

		# Prepare blob data for API
		blob_data = []
		if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
			for _thread_id_b, _checkpoint_ns_b, channel, version, type_str, blob in self._dump_blobs(
				thread_id, checkpoint_ns, blob_values, blob_versions
			):
				blob_data.append(
					{
						"channel": channel,
						"version": version,
						"type": type_str,
						"blob": encode_to_base64(blob) if blob is not None else None,
					}
				)

		# Create request using Pydantic model
		checkpoint_data = CheckpointData(
			v=copy.get("v", 1),
			ts=copy["ts"],
			id=copy["id"],
			channel_values=copy.get("channel_values", {}),
			channel_versions=copy.get("channel_versions", {}),
			versions_seen=copy.get("versions_seen"),
		)

		# Create BlobData list if needed
		blob_data_models = [BlobData(**bd) for bd in blob_data] if blob_data else None

		request_model = CheckpointRequest(
			thread_id=thread_id,
			checkpoint_ns=checkpoint_ns,
			checkpoint_id=checkpoint["id"],
			parent_checkpoint_id=parent_checkpoint_id,
			checkpoint=checkpoint_data,
			metadata=get_checkpoint_metadata(config, metadata),
			blob_data=blob_data_models,
		)
		request = request_model.model_dump(exclude_none=True)

		with self.lock, self._get_client() as client:
			try:
				# Extract thread_id and checkpoint_id from request for URL
				thread_id_str = request["thread_id"]
				checkpoint_id_str = request["checkpoint_id"]
				path = f"/checkpoints/{thread_id_str}/{checkpoint_id_str}"
				client.put(path, request)
			except HTTPClientError:
				raise

		return {
			"configurable": {
				"thread_id": thread_id,
				"checkpoint_ns": checkpoint_ns,
				"checkpoint_id": checkpoint["id"],
			}
		}

	def put_writes(
		self,
		config: RunnableConfig,
		writes: Sequence[tuple[str, Any]],
		task_id: str,
		task_path: str = "",
	) -> None:
		"""Store intermediate writes linked to a checkpoint via HTTP API."""
		thread_id = config["configurable"]["thread_id"]
		checkpoint_ns = config["configurable"]["checkpoint_ns"]
		checkpoint_id = config["configurable"]["checkpoint_id"]

		# Prepare write data using Pydantic models
		write_data_models = []
		for idx, (channel, value) in enumerate(writes):
			type_str, blob = self.serde.dumps_typed(value)
			write_data_models.append(
				WriteData(
					idx=WRITES_IDX_MAP.get(channel, idx),
					channel=channel,
					type=type_str,
					blob=encode_to_base64(blob),
				)
			)

		request_model = CheckpointWriteRequest(
			thread_id=thread_id,
			checkpoint_ns=checkpoint_ns,
			checkpoint_id=checkpoint_id,
			task_id=str(task_id),  # Ensure task_id is string
			task_path=task_path,
			writes=write_data_models,
		)
		request = request_model.model_dump()

		with self.lock, self._get_client() as client:
			try:
				# Extract thread_id and checkpoint_id from request for URL
				thread_id_str = request["thread_id"]
				checkpoint_id_str = request["checkpoint_id"]
				path = f"/checkpoints/{thread_id_str}/{checkpoint_id_str}/writes"
				client.put(path, request)
			except HTTPClientError:
				raise

	def delete_thread(self, thread_id: str) -> None:
		"""Delete all checkpoints and writes associated with a thread ID via HTTP API."""
		with self.lock, self._get_client() as client:
			try:
				response = client.delete(f"/checkpoints/{thread_id}")
				parsed = ThreadDeleteResponse(**response)
				if not parsed.success:
					raise HTTPClientError("Delete thread failed")
			except HTTPClientError:
				raise

	def _checkpoint_response_to_tuple(self, response: dict[str, Any]) -> CheckpointTuple:
		"""Convert API response to CheckpointTuple."""
		# Use utility functions to transform the data
		channel_values_parsed = transform_channel_values(response.get("channel_values"))
		pending_writes_parsed = transform_pending_writes(response.get("pending_writes"))

		# Build checkpoint tuple
		checkpoint_data = response["checkpoint"]
		if isinstance(checkpoint_data, str):
			checkpoint_data = json.loads(checkpoint_data)

		return CheckpointTuple(
			{
				"configurable": {
					"thread_id": response["thread_id"],
					"checkpoint_ns": response["checkpoint_ns"],
					"checkpoint_id": response["checkpoint_id"],
				}
			},
			{
				**checkpoint_data,
				"channel_values": {
					**checkpoint_data.get("channel_values", {}),
					**self._load_blobs(channel_values_parsed),
				},
			},
			response["metadata"],
			(
				{
					"configurable": {
						"thread_id": response["thread_id"],
						"checkpoint_ns": response["checkpoint_ns"],
						"checkpoint_id": response["parent_checkpoint_id"],
					}
				}
				if response.get("parent_checkpoint_id")
				else None
			),
			self._load_writes(pending_writes_parsed),
		)


__all__ = [
	"INVALID_REQUEST_PAYLOAD",
	"INVALID_RESPONSE",
	"HTTPClient",
	"HTTPClientError",
	"HTTPSingleStoreSaver",
	"RetryConfig",
]
