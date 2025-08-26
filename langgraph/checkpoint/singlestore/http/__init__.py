"""HTTP-based SingleStore checkpointer implementation."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig

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
from .models import (
	CheckpointListRequest,
	CheckpointRequest,
	CheckpointResponse,
	CheckpointWriteRequest,
	WriteData,
)


class HTTPSingleStoreSaver(BaseSingleStoreSaver):
	"""HTTP-based checkpointer that stores checkpoints via API calls."""

	lock: threading.Lock

	def __init__(
		self,
		base_url: str,
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
			api_key: Optional API key for authentication
			serde: Serializer for checkpoint data
			timeout: Request timeout in seconds
			retry_config: Configuration for request retries
			pool_connections: Number of connection pool connections
			pool_maxsize: Maximum size of connection pool
		"""
		super().__init__(serde=serde)
		self.base_url = base_url
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
		api_key: str | None = None,
		**kwargs,
	) -> Iterator[HTTPSingleStoreSaver]:
		"""Create a new HTTPSingleStoreSaver instance from a URL.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			api_key: Optional API key for authentication
			**kwargs: Additional arguments for HTTPSingleStoreSaver

		Yields:
			HTTPSingleStoreSaver instance
		"""
		saver = cls(base_url=base_url, api_key=api_key, **kwargs)
		client = HTTPClient(
			base_url=base_url,
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
			response = client.post("/setup", {})
			if not response.get("success"):
				raise HTTPClientError(f"Setup failed: {response.get('message', 'Unknown error')}")

	def list(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> Iterator[CheckpointTuple]:
		"""List checkpoints from the database via HTTP API."""
		request_data: CheckpointListRequest = {}

		if config:
			request_data["thread_id"] = config["configurable"]["thread_id"]
			checkpoint_ns = config["configurable"].get("checkpoint_ns")
			if checkpoint_ns is not None:
				request_data["checkpoint_ns"] = checkpoint_ns
			if checkpoint_id := get_checkpoint_id(config):
				request_data["checkpoint_id"] = checkpoint_id

		if filter:
			request_data["metadata_filter"] = filter

		if before:
			request_data["before_checkpoint_id"] = get_checkpoint_id(before)

		if limit:
			request_data["limit"] = limit

		with self._get_client() as client:
			response = client.get("/checkpoints", params=request_data)
			checkpoints = response.get("checkpoints", [])

			for checkpoint_data in checkpoints:
				yield self._checkpoint_response_to_tuple(checkpoint_data)

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
					path = f"/checkpoints/{thread_id}/{checkpoint_ns}/{checkpoint_id}"
				else:
					# Get latest checkpoint
					path = f"/checkpoints/{thread_id}/{checkpoint_ns}/latest"

				response = client.get(path)
				return self._checkpoint_response_to_tuple(response)
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
						"blob": self._encode_blob(blob) if blob else None,
					}
				)

		# Create request
		request: CheckpointRequest = {
			"thread_id": thread_id,
			"checkpoint_ns": checkpoint_ns,
			"checkpoint_id": checkpoint["id"],
			"parent_checkpoint_id": parent_checkpoint_id,
			"checkpoint": copy,
			"metadata": get_checkpoint_metadata(config, metadata),
			"blob_data": blob_data if blob_data else None,
		}

		with self.lock, self._get_client() as client:
			client.put("/checkpoints", request)

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

		# Prepare write data
		write_data: list[WriteData] = []
		for idx, (channel, value) in enumerate(writes):
			type_str, blob = self.serde.dumps_typed(value)
			write_data.append(
				{
					"idx": WRITES_IDX_MAP.get(channel, idx),
					"channel": channel,
					"type": type_str,
					"blob": self._encode_blob(blob),
				}
			)

		request: CheckpointWriteRequest = {
			"thread_id": thread_id,
			"checkpoint_ns": checkpoint_ns,
			"checkpoint_id": checkpoint_id,
			"task_id": task_id,
			"task_path": task_path,
			"writes": write_data,
		}

		with self.lock, self._get_client() as client:
			client.put("/checkpoint-writes", request)

	def delete_thread(self, thread_id: str) -> None:
		"""Delete all checkpoints and writes associated with a thread ID via HTTP API."""
		with self.lock, self._get_client() as client:
			client.delete(f"/threads/{thread_id}")

	def _encode_blob(self, data: bytes) -> str:
		"""Encode binary data to base64 string."""
		import base64

		return base64.b64encode(data).decode("utf-8")

	def _decode_blob(self, data: str) -> bytes:
		"""Decode base64 string to binary data."""
		import base64

		return base64.b64decode(data)

	def _checkpoint_response_to_tuple(self, response: CheckpointResponse) -> CheckpointTuple:
		"""Convert API response to CheckpointTuple."""
		# Parse channel values from response
		channel_values_parsed = []
		if response.get("channel_values"):
			for item in response["channel_values"]:
				channel_values_parsed.append(
					(
						item[0].encode("utf-8"),  # channel
						item[1].encode("utf-8"),  # type
						self._decode_blob(item[2]) if item[2] else b"",  # blob
					)
				)

		# Parse pending writes from response
		pending_writes_parsed = []
		if response.get("pending_writes"):
			for item in response["pending_writes"]:
				pending_writes_parsed.append(
					(
						item[0].encode("utf-8"),  # task_id
						item[1].encode("utf-8"),  # channel
						item[2].encode("utf-8"),  # type
						self._decode_blob(item[3]) if item[3] else b"",  # blob
					)
				)

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


__all__ = ["HTTPClient", "HTTPClientError", "HTTPSingleStoreSaver", "RetryConfig"]
