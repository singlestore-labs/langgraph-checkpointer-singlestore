"""Async HTTP-based SingleStore checkpointer implementation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
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

from .client import AsyncHTTPClient, HTTPClientError, RetryConfig
from .models import (
	CheckpointListRequest,
	CheckpointRequest,
	CheckpointResponse,
	CheckpointWriteRequest,
	WriteData,
)


class AsyncHTTPSingleStoreSaver(BaseSingleStoreSaver):
	"""Async HTTP-based checkpointer that stores checkpoints via API calls."""

	lock: asyncio.Lock

	def __init__(
		self,
		base_url: str,
		api_key: str | None = None,
		serde: SerializerProtocol | None = None,
		timeout: float = 30.0,
		retry_config: RetryConfig | None = None,
		max_connections: int = 100,
		max_keepalive_connections: int = 20,
	) -> None:
		"""Initialize async HTTP-based SingleStore checkpointer.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			api_key: Optional API key for authentication
			serde: Serializer for checkpoint data
			timeout: Request timeout in seconds
			retry_config: Configuration for request retries
			max_connections: Maximum number of connections
			max_keepalive_connections: Maximum keepalive connections
		"""
		super().__init__(serde=serde)
		self.base_url = base_url
		self.api_key = api_key
		self.timeout = timeout
		self.retry_config = retry_config
		self.max_connections = max_connections
		self.max_keepalive_connections = max_keepalive_connections
		self.lock = asyncio.Lock()
		self.loop = asyncio.get_running_loop()
		self._client: AsyncHTTPClient | None = None

	@asynccontextmanager
	async def _get_client(self) -> AsyncIterator[AsyncHTTPClient]:
		"""Get or create async HTTP client."""
		if self._client is None:
			client = AsyncHTTPClient(
				base_url=self.base_url,
				api_key=self.api_key,
				timeout=self.timeout,
				retry_config=self.retry_config,
				max_connections=self.max_connections,
				max_keepalive_connections=self.max_keepalive_connections,
			)
			async with client.create() as http_client:
				yield http_client
		else:
			yield self._client

	@classmethod
	@asynccontextmanager
	async def from_url(
		cls,
		base_url: str,
		api_key: str | None = None,
		**kwargs,
	) -> AsyncIterator[AsyncHTTPSingleStoreSaver]:
		"""Create a new AsyncHTTPSingleStoreSaver instance from a URL.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			api_key: Optional API key for authentication
			**kwargs: Additional arguments for AsyncHTTPSingleStoreSaver

		Yields:
			AsyncHTTPSingleStoreSaver instance
		"""
		saver = cls(base_url=base_url, api_key=api_key, **kwargs)
		client = AsyncHTTPClient(
			base_url=base_url,
			api_key=api_key,
			timeout=kwargs.get("timeout", 30.0),
			retry_config=kwargs.get("retry_config"),
			max_connections=kwargs.get("max_connections", 100),
			max_keepalive_connections=kwargs.get("max_keepalive_connections", 20),
		)
		async with client.create() as http_client:
			saver._client = http_client
			try:
				yield saver
			finally:
				saver._client = None

	async def setup(self) -> None:
		"""Set up the checkpoint database via HTTP API."""
		async with self.lock, self._get_client() as client:
			response = await client.post("/setup", {})
			if not response.get("success"):
				raise HTTPClientError(f"Setup failed: {response.get('message', 'Unknown error')}")

	async def alist(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> AsyncIterator[CheckpointTuple]:
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

		async with self._get_client() as client:
			response = await client.get("/checkpoints", params=request_data)
			checkpoints = response.get("checkpoints", [])

			for checkpoint_data in checkpoints:
				yield await self._checkpoint_response_to_tuple(checkpoint_data)

	async def aget(self, config: RunnableConfig) -> Checkpoint | None:
		"""Get a checkpoint from the database via HTTP API."""
		checkpoint_tuple = await self.aget_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
		"""Get a checkpoint tuple from the database via HTTP API."""
		thread_id = config["configurable"]["thread_id"]
		checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
		checkpoint_id = get_checkpoint_id(config)

		async with self._get_client() as client:
			try:
				if checkpoint_id:
					# Get specific checkpoint
					path = f"/checkpoints/{thread_id}/{checkpoint_ns}/{checkpoint_id}"
				else:
					# Get latest checkpoint
					path = f"/checkpoints/{thread_id}/{checkpoint_ns}/latest"

				response = await client.get(path)
				return await self._checkpoint_response_to_tuple(response)
			except HTTPClientError as e:
				if e.status_code == 404:
					return None
				raise

	async def aput(
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
			blob_list = await asyncio.to_thread(
				self._dump_blobs,
				thread_id,
				checkpoint_ns,
				blob_values,
				blob_versions,
			)
			for _thread_id_b, _checkpoint_ns_b, channel, version, type_str, blob in blob_list:
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

		async with self.lock, self._get_client() as client:
			await client.put("/checkpoints", request)

		return {
			"configurable": {
				"thread_id": thread_id,
				"checkpoint_ns": checkpoint_ns,
				"checkpoint_id": checkpoint["id"],
			}
		}

	async def aput_writes(
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

		async with self.lock, self._get_client() as client:
			await client.put("/checkpoint-writes", request)

	async def adelete_thread(self, thread_id: str) -> None:
		"""Delete all checkpoints and writes associated with a thread ID via HTTP API."""
		async with self.lock, self._get_client() as client:
			await client.delete(f"/threads/{thread_id}")

	def _encode_blob(self, data: bytes) -> str:
		"""Encode binary data to base64 string."""
		import base64

		return base64.b64encode(data).decode("utf-8")

	def _decode_blob(self, data: str) -> bytes:
		"""Decode base64 string to binary data."""
		import base64

		return base64.b64decode(data)

	async def _checkpoint_response_to_tuple(self, response: CheckpointResponse) -> CheckpointTuple:
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

	# Sync methods that delegate to async via run_coroutine_threadsafe
	def list(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> Iterator[CheckpointTuple]:
		"""List checkpoints (sync bridge to async)."""
		try:
			if asyncio.get_running_loop() is self.loop:
				raise asyncio.InvalidStateError(
					"Synchronous calls to AsyncHTTPSingleStoreSaver are only allowed from a "
					"different thread. From the main thread, use the async interface. "
					"For example, use `checkpointer.alist(...)` or `await graph.ainvoke(...)`."
				)
		except RuntimeError:
			pass

		aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
		while True:
			try:
				yield asyncio.run_coroutine_threadsafe(
					anext(aiter_),  # type: ignore[arg-type]
					self.loop,
				).result()
			except StopAsyncIteration:
				break

	def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
		"""Get checkpoint tuple (sync bridge to async)."""
		try:
			if asyncio.get_running_loop() is self.loop:
				raise asyncio.InvalidStateError(
					"Synchronous calls to AsyncHTTPSingleStoreSaver are only allowed from a "
					"different thread. From the main thread, use the async interface."
				)
		except RuntimeError:
			pass
		return asyncio.run_coroutine_threadsafe(self.aget_tuple(config), self.loop).result()

	def get(self, config: RunnableConfig) -> Checkpoint | None:
		"""Get checkpoint (sync bridge to async)."""
		checkpoint_tuple = self.get_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	def put(
		self,
		config: RunnableConfig,
		checkpoint: Checkpoint,
		metadata: CheckpointMetadata,
		new_versions: ChannelVersions,
	) -> RunnableConfig:
		"""Save checkpoint (sync bridge to async)."""
		return asyncio.run_coroutine_threadsafe(
			self.aput(config, checkpoint, metadata, new_versions), self.loop
		).result()

	def put_writes(
		self,
		config: RunnableConfig,
		writes: Sequence[tuple[str, Any]],
		task_id: str,
		task_path: str = "",
	) -> None:
		"""Store writes (sync bridge to async)."""
		return asyncio.run_coroutine_threadsafe(
			self.aput_writes(config, writes, task_id, task_path), self.loop
		).result()

	def delete_thread(self, thread_id: str) -> None:
		"""Delete thread (sync bridge to async)."""
		try:
			if asyncio.get_running_loop() is self.loop:
				raise asyncio.InvalidStateError(
					"Synchronous calls to AsyncHTTPSingleStoreSaver are only allowed from a "
					"different thread. From the main thread, use the async interface."
				)
		except RuntimeError:
			pass
		return asyncio.run_coroutine_threadsafe(self.adelete_thread(thread_id), self.loop).result()


__all__ = ["AsyncHTTPSingleStoreSaver"]
