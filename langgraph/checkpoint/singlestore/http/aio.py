"""Async HTTP-based SingleStore checkpointer implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import pprint
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, suppress
from functools import wraps
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

from .client import AsyncHTTPClient, HTTPClientError, RetryConfig
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
	TokenGetter,
	encode_to_base64,
	prepare_metadata_filter,
	transform_channel_values,
	transform_pending_writes,
)

# Configure logger for this module
logger = logging.getLogger(__name__)


def _make_readable(obj: Any, max_str_length: int = 200) -> str:
	"""Convert objects to human-readable format with proper indentation."""
	if obj is None:
		return "None"

	if isinstance(obj, str | int | float | bool):
		if isinstance(obj, str) and len(obj) > max_str_length:
			return f'"{obj[:max_str_length]}..."'
		return repr(obj)

	if isinstance(obj, list | tuple | dict):
		# Use pprint for complex structures
		formatted = pprint.pformat(obj, indent=2, width=120, depth=4)
		# Limit total output length
		if len(formatted) > 1000:
			return formatted[:1000] + "..."
		return formatted

	# For Pydantic models and other objects with dict representation
	if hasattr(obj, "model_dump"):
		return _make_readable(obj.model_dump())
	elif hasattr(obj, "__dict__"):
		return _make_readable(obj.__dict__)
	else:
		return str(obj)


def log_method_async(func):
	"""Decorator to log async method inputs and outputs with nice formatting."""

	@wraps(func)
	async def wrapper(self, *args, **kwargs):
		method_name = func.__name__

		# Log method entry
		logger.info("")  # Empty line for spacing
		logger.info(f"{'=' * 80}")
		logger.info(f"ENTERING: {method_name}")
		logger.info(f"{'=' * 80}")

		# Log inputs
		if args:
			logger.info("ARGS:")
			for i, arg in enumerate(args):
				logger.info(f"  [{i}]: {_make_readable(arg)}")

		if kwargs:
			logger.info("KWARGS:")
			for key, value in kwargs.items():
				logger.info(f"  {key}: {_make_readable(value)}")

		logger.info("")  # Empty line after inputs

		try:
			# Execute the method
			result = await func(self, *args, **kwargs)

			# Log output
			logger.info(f"{'=' * 80}")
			logger.info(f"EXITING: {method_name}")
			logger.info(f"{'=' * 80}")
			logger.info(f"RESULT: {_make_readable(result)}")
			logger.info("")  # Empty line after result

			return result

		except Exception as e:
			# Log exceptions
			logger.error(f"{'=' * 80}")
			logger.error(f"ERROR in {method_name}")
			logger.error(f"{'=' * 80}")
			logger.error(f"Exception: {type(e).__name__}: {e!s}")
			logger.error("")  # Empty line after error
			raise

	return wrapper


def log_method_sync(func):
	"""Decorator to log sync method inputs and outputs with nice formatting."""

	@wraps(func)
	def wrapper(self, *args, **kwargs):
		method_name = func.__name__

		# Log method entry
		logger.info("")  # Empty line for spacing
		logger.info(f"{'=' * 80}")
		logger.info(f"ENTERING: {method_name}")
		logger.info(f"{'=' * 80}")

		# Log inputs
		if args:
			logger.info("ARGS:")
			for i, arg in enumerate(args):
				logger.info(f"  [{i}]: {_make_readable(arg)}")

		if kwargs:
			logger.info("KWARGS:")
			for key, value in kwargs.items():
				logger.info(f"  {key}: {_make_readable(value)}")

		logger.info("")  # Empty line after inputs

		try:
			# Execute the method
			result = func(self, *args, **kwargs)

			# Log output
			logger.info(f"{'=' * 80}")
			logger.info(f"EXITING: {method_name}")
			logger.info(f"{'=' * 80}")
			logger.info(f"RESULT: {_make_readable(result)}")
			logger.info("")  # Empty line after result

			return result

		except Exception as e:
			# Log exceptions
			logger.error(f"{'=' * 80}")
			logger.error(f"ERROR in {method_name}")
			logger.error(f"{'=' * 80}")
			logger.error(f"Exception: {type(e).__name__}: {e!s}")
			logger.error("")  # Empty line after error
			raise

	return wrapper


def log_async_iterator(func):
	"""Special decorator for async methods that return AsyncIterator."""

	@wraps(func)
	async def wrapper(self, *args, **kwargs):
		method_name = func.__name__

		# Log method entry
		logger.info("")  # Empty line for spacing
		logger.info(f"{'=' * 80}")
		logger.info(f"ENTERING: {method_name}")
		logger.info(f"{'=' * 80}")

		# Log inputs
		if args:
			logger.info("ARGS:")
			for i, arg in enumerate(args):
				logger.info(f"  [{i}]: {_make_readable(arg)}")

		if kwargs:
			logger.info("KWARGS:")
			for key, value in kwargs.items():
				logger.info(f"  {key}: {_make_readable(value)}")

		logger.info("")  # Empty line after inputs

		try:
			# Execute the method to get the async iterator
			async_iter = func(self, *args, **kwargs)

			# Log that we're returning an iterator
			logger.info(f"{'=' * 80}")
			logger.info(f"CREATED ASYNC ITERATOR: {method_name}")
			logger.info(f"{'=' * 80}")
			logger.info("")

			# Wrap the iterator to log yielded values
			async def logged_iterator():
				item_count = 0
				async for item in async_iter:
					item_count += 1
					logger.info(f"{'=' * 80}")
					logger.info(f"YIELDING ITEM {item_count} from {method_name}")
					logger.info(f"{'=' * 80}")
					logger.info(f"ITEM: {_make_readable(item)}")
					logger.info("")
					yield item

				logger.info(f"{'=' * 80}")
				logger.info(f"ITERATOR EXHAUSTED: {method_name} (yielded {item_count} items)")
				logger.info(f"{'=' * 80}")
				logger.info("")

			return logged_iterator()

		except Exception as e:
			# Log exceptions
			logger.error(f"{'=' * 80}")
			logger.error(f"ERROR in {method_name}")
			logger.error(f"{'=' * 80}")
			logger.error(f"Exception: {type(e).__name__}: {e!s}")
			logger.error("")  # Empty line after error
			raise

	return wrapper


class AsyncHTTPSingleStoreSaver(BaseSingleStoreSaver):
	"""Async HTTP-based checkpointer that stores checkpoints via API calls."""

	lock: asyncio.Lock

	# @log_method_sync
	def __init__(
		self,
		base_url: str,
		api_key: TokenGetter,
		base_path: str = "",
		serde: SerializerProtocol | None = None,
		timeout: float = 30.0,
		retry_config: RetryConfig | None = None,
		max_connections: int = 100,
		max_keepalive_connections: int = 20,
	) -> None:
		"""Initialize async HTTP-based SingleStore checkpointer.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			base_path: Base path to prepend to all endpoints
			api_key: Optional API key for authentication
			serde: Serializer for checkpoint data
			timeout: Request timeout in seconds
			retry_config: Configuration for request retries
			max_connections: Maximum number of connections
			max_keepalive_connections: Maximum keepalive connections
		"""
		super().__init__(serde=serde)
		self.base_url = base_url
		self.base_path = base_path
		self.api_key = api_key
		self.timeout = timeout
		self.retry_config = retry_config
		self.max_connections = max_connections
		self.max_keepalive_connections = max_keepalive_connections
		self.lock = asyncio.Lock()
		self._loop: asyncio.AbstractEventLoop | None = None
		self._client: AsyncHTTPClient | None = None
		self._client_context = None

	@property
	def loop(self) -> asyncio.AbstractEventLoop:
		"""Get the event loop lazily."""
		if self._loop is None:
			try:
				self._loop = asyncio.get_running_loop()
			except RuntimeError:
				# No running loop, get the event loop
				self._loop = asyncio.get_event_loop()
				if self._loop is None:
					self._loop = asyncio.new_event_loop()
					asyncio.set_event_loop(self._loop)
		return self._loop

	async def open(self) -> AsyncHTTPSingleStoreSaver:
		"""Open the async HTTP client connection.

		Creates and stores the HTTP client for connection pooling.
		Can be called multiple times safely.

		Returns:
			Self for method chaining
		"""
		if self._client is None:
			client = AsyncHTTPClient(
				base_url=self.base_url,
				api_key_getter=self.api_key,
				base_path=self.base_path,
				timeout=self.timeout,
				retry_config=self.retry_config,
				max_connections=self.max_connections,
				max_keepalive_connections=self.max_keepalive_connections,
			)
			# Create the actual httpx async client
			self._client_context = client.create()
			self._client = await self._client_context.__aenter__()
		return self

	async def close(self) -> None:
		"""Close the async HTTP client connection.

		Cleans up the HTTP client and connection pool.
		"""
		if self._client is not None:
			if hasattr(self, "_client_context") and self._client_context:
				with suppress(Exception):
					await self._client_context.__aexit__(None, None, None)
			self._client = None
			self._client_context = None

	async def __aenter__(self) -> AsyncHTTPSingleStoreSaver:
		"""Enter async context manager."""
		return await self.open()

	async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
		"""Exit async context manager."""
		await self.close()

	@asynccontextmanager
	async def _get_client(self) -> AsyncIterator[AsyncHTTPClient]:
		"""Get or create async HTTP client.

		Lazily creates and caches the client on first use.
		"""
		if self._client is None:
			await self.open()
		yield self._client

	@classmethod
	@asynccontextmanager
	async def from_url(
		cls,
		base_url: str,
		api_key: TokenGetter,
		base_path: str = "",
		**kwargs,
	) -> AsyncIterator[AsyncHTTPSingleStoreSaver]:
		"""Create a new AsyncHTTPSingleStoreSaver instance from a URL.

		Args:
			base_url: Base URL of the checkpoint HTTP server
			base_path: Base path to prepend to all endpoints
			api_key: Optional API key getter for authentication
			**kwargs: Additional arguments for AsyncHTTPSingleStoreSaver

		Yields:
			AsyncHTTPSingleStoreSaver instance
		"""
		saver = cls(base_url=base_url, api_key=api_key, base_path=base_path, **kwargs)
		try:
			await saver.open()
			yield saver
		finally:
			await saver.close()

	# @log_method_async
	async def setup(self) -> None:
		"""Set up the checkpoint database via HTTP API."""
		async with self.lock, self._get_client() as client:
			try:
				response = await client.post("/checkpoints/setup", {})

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

	# @log_async_iterator
	async def alist(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> AsyncIterator[CheckpointTuple]:
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

		async with self._get_client() as client:
			try:
				response = await client.get("/checkpoints", params=request_data)

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

	# @log_method_async
	async def aget(self, config: RunnableConfig) -> Checkpoint | None:
		"""Get a checkpoint from the database via HTTP API."""
		checkpoint_tuple = await self.aget_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	# @log_method_async
	async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
		"""Get a checkpoint tuple from the database via HTTP API."""
		thread_id = config["configurable"]["thread_id"]
		checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
		checkpoint_id = get_checkpoint_id(config)

		async with self._get_client() as client:
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

				response = await client.get(path, params=params)

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

	# @log_method_async
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

		async with self.lock, self._get_client() as client:
			try:
				# Extract thread_id and checkpoint_id from request for URL
				thread_id_str = request["thread_id"]
				checkpoint_id_str = request["checkpoint_id"]
				path = f"/checkpoints/{thread_id_str}/{checkpoint_id_str}"
				await client.put(path, request)
			except HTTPClientError:
				raise

		return {
			"configurable": {
				"thread_id": thread_id,
				"checkpoint_ns": checkpoint_ns,
				"checkpoint_id": checkpoint["id"],
			}
		}

	# @log_method_async
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

		async with self.lock, self._get_client() as client:
			try:
				# Extract thread_id and checkpoint_id from request for URL
				thread_id_str = request["thread_id"]
				checkpoint_id_str = request["checkpoint_id"]
				path = f"/checkpoints/{thread_id_str}/{checkpoint_id_str}/writes"
				await client.put(path, request)
			except HTTPClientError:
				raise

	# @log_method_async
	async def adelete_thread(self, thread_id: str) -> None:
		"""Delete all checkpoints and writes associated with a thread ID via HTTP API."""
		async with self.lock, self._get_client() as client:
			try:
				response = await client.delete(f"/checkpoints/{thread_id}")
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

	# Sync methods that delegate to async via run_coroutine_threadsafe
	# @log_method_sync
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

	# @log_method_sync
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

	# @log_method_sync
	def get(self, config: RunnableConfig) -> Checkpoint | None:
		"""Get checkpoint (sync bridge to async)."""
		checkpoint_tuple = self.get_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	# @log_method_sync
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

	# @log_method_sync
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

	# @log_method_sync
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
