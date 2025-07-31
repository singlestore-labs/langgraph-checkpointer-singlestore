from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import singlestoredb
from langchain_core.runnables import RunnableConfig
from singlestoredb.connection import Connection, Cursor

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
from langgraph.checkpoint.singlestore import _ainternal
from langgraph.checkpoint.singlestore.base import BaseSingleStoreSaver


class AsyncSingleStoreSaver(BaseSingleStoreSaver):
	"""Asynchronous checkpointer that stores checkpoints in a SingleStore database."""

	lock: asyncio.Lock

	def __init__(
		self,
		conn: Connection,
		serde: SerializerProtocol | None = None,
	) -> None:
		super().__init__(serde=serde)
		self.conn = conn
		self.lock = asyncio.Lock()
		self.loop = asyncio.get_running_loop()

	@classmethod
	@asynccontextmanager
	async def from_conn_string(
		cls,
		conn_string: str,
		*,
		serde: SerializerProtocol | None = None,
	) -> AsyncIterator[AsyncSingleStoreSaver]:
		"""Create a new AsyncSingleStoreSaver instance from a connection string.

		Args:
		conn_string: The SingleStore connection string.
		serde: Serializer protocol for data serialization.

		Returns:
		AsyncSingleStoreSaver: A new AsyncSingleStoreSaver instance.
		"""
		with singlestoredb.connect(conn_string, autocommit=True, results_type="dict") as conn:
			yield cls(conn=conn, serde=serde)

	async def setup(self) -> None:
		"""Set up the checkpoint database asynchronously.

		This method creates the necessary tables in the SingleStore database if they don't
		already exist and runs database migrations. It MUST be called directly by the user
		the first time checkpointer is used.
		"""
		async with self._cursor() as cur:
			await asyncio.to_thread(cur.execute, self.MIGRATIONS[0])
			await asyncio.to_thread(cur.execute, "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
			row = await asyncio.to_thread(cur.fetchone)
			version = -1 if row is None else row["v"]
			for v, migration in zip(
				range(version + 1, len(self.MIGRATIONS)),
				self.MIGRATIONS[version + 1 :],
				strict=False,
			):
				try:
					await asyncio.to_thread(cur.execute, migration)
					await asyncio.to_thread(cur.execute, f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")
				except Exception as e:
					print(f"Error applying migration {migration}: {e}")
					raise e

	async def alist(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> AsyncIterator[CheckpointTuple]:
		"""List checkpoints from the database asynchronously.

		This method retrieves a list of checkpoint tuples from the SingleStore database based
		on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

		Args:
		config: Base configuration for filtering checkpoints.
		filter: Additional filtering criteria for metadata.
		before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
		limit: Maximum number of checkpoints to return.

		Yields:
		AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
		"""
		where, args = self._search_where(config, filter, before)
		query = self.SELECT_SQL.replace("{{where}}", where) + " ORDER BY c.checkpoint_id DESC"
		if limit:
			query += f" LIMIT {limit}"
		query += ";"

		async with self._cursor() as cur:
			await asyncio.to_thread(cur.execute, query, args)
			values = await asyncio.to_thread(cur.fetchall)
			if not values:
				return
			# migrate pending sends if necessary
			if to_migrate := [v for v in values if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]]:
				placeholders = ", ".join(["%s"] * len([v["parent_checkpoint_id"] for v in to_migrate]))
				pending_sends_query = self.SELECT_PENDING_SENDS_SQL.replace("IN (%s)", f"IN ({placeholders})")
				await asyncio.to_thread(
					cur.execute,
					pending_sends_query,
					(values[0]["thread_id"], *[v["parent_checkpoint_id"] for v in to_migrate]),
				)
				grouped_by_parent = defaultdict(list)
				for value in to_migrate:
					grouped_by_parent[value["parent_checkpoint_id"]].append(value)
				sends_results = await asyncio.to_thread(cur.fetchall)
				for sends in sends_results:
					for value in grouped_by_parent[sends["checkpoint_id"]]:
						# Parse channel_values if it's a JSON string
						channel_values = value.get("channel_values")
						if channel_values is None:
							channel_values = []
						elif isinstance(channel_values, str):
							import json

							try:
								channel_values = json.loads(channel_values)
							except json.JSONDecodeError:
								channel_values = []

						self._migrate_pending_sends(
							sends["sends"],
							value["checkpoint"],
							channel_values,
						)
						# Update the checkpoint's channel_values with the migrated data
						if channel_values:
							value["checkpoint"]["channel_values"] = self._load_blobs(channel_values)
			for value in values:
				yield await self._load_checkpoint_tuple(value)

	async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
		"""Get a checkpoint tuple from the database asynchronously.

		This method retrieves a checkpoint tuple from the SingleStore database based on the
		provided config. If the config contains a "checkpoint_id" key, the checkpoint with
		the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
		for the given thread ID is retrieved.

		Args:
		config: The config to use for retrieving the checkpoint.

		Returns:
		Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
		"""
		thread_id = config["configurable"]["thread_id"]
		checkpoint_id = get_checkpoint_id(config)
		checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

		async with self._cursor() as cur:
			if checkpoint_id:
				# Specific checkpoint query
				args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
				where = "WHERE c.thread_id = %s AND c.checkpoint_ns = %s AND c.checkpoint_id = %s"
				query = self.SELECT_SQL.replace("{{where}}", where) + ";"
			else:
				# Latest checkpoint query - need to handle ORDER BY and LIMIT properly
				args = (thread_id, checkpoint_ns)
				where = "WHERE c.thread_id = %s AND c.checkpoint_ns = %s"
				query = self.SELECT_SQL.replace("{{where}}", where) + " ORDER BY c.checkpoint_id DESC LIMIT 1;"

			try:
				await asyncio.to_thread(cur.execute, query, args)
				value = await asyncio.to_thread(cur.fetchone)
				if value is None:
					return None

				# migrate pending sends if necessary
				if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
					await asyncio.to_thread(
						cur.execute,
						self.SELECT_PENDING_SENDS_SQL,
						(thread_id, value["parent_checkpoint_id"]),
					)
					sends = await asyncio.to_thread(cur.fetchone)
					if sends:
						# Parse channel_values if it's a JSON string
						channel_values = value.get("channel_values")
						if channel_values is None:
							channel_values = []
						elif isinstance(channel_values, str):
							import json

							try:
								channel_values = json.loads(channel_values)
							except json.JSONDecodeError:
								channel_values = []

						self._migrate_pending_sends(
							sends["sends"],
							value["checkpoint"],
							channel_values,
						)
						# Update the checkpoint's channel_values with the migrated data
						if channel_values:
							value["checkpoint"]["channel_values"] = self._load_blobs(channel_values)

				return await self._load_checkpoint_tuple(value)
			except Exception as e:
				print(f"Error executing query: {e}")
				print(f"Query length: {len(query)}")
				print(f"Query ends with: {query[-50:] if len(query) > 50 else query}")
				raise

	async def aput(
		self,
		config: RunnableConfig,
		checkpoint: Checkpoint,
		metadata: CheckpointMetadata,
		new_versions: ChannelVersions,
	) -> RunnableConfig:
		"""Save a checkpoint to the database asynchronously.

		This method saves a checkpoint to the SingleStore database. The checkpoint is associated
		with the provided config and its parent config (if any).

		Args:
		config: The config to associate with the checkpoint.
		checkpoint: The checkpoint to save.
		metadata: Additional metadata to save with the checkpoint.
		new_versions: New channel versions as of this write.

		Returns:
		RunnableConfig: Updated configuration after storing the checkpoint.
		"""
		configurable = config["configurable"].copy()
		thread_id = configurable.pop("thread_id")
		checkpoint_ns = configurable.pop("checkpoint_ns")
		checkpoint_id = configurable.pop("checkpoint_id", None)
		copy = checkpoint.copy()
		copy["channel_values"] = copy["channel_values"].copy()
		next_config = {
			"configurable": {
				"thread_id": thread_id,
				"checkpoint_ns": checkpoint_ns,
				"checkpoint_id": checkpoint["id"],
			}
		}

		# inline primitive values in checkpoint table
		# others are stored in blobs table
		blob_values = {}
		for k, v in checkpoint["channel_values"].items():
			if v is None or isinstance(v, str | int | float | bool):
				pass
			else:
				blob_values[k] = copy["channel_values"].pop(k)

		async with self._cursor() as cur:
			if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
				blob_data_list = await asyncio.to_thread(
					self._dump_blobs,
					thread_id,
					checkpoint_ns,
					blob_values,
					blob_versions,
				)
				for blob_data in blob_data_list:
					await asyncio.to_thread(cur.execute, self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_data)

			import json

			await asyncio.to_thread(
				cur.execute,
				self.UPSERT_CHECKPOINTS_SQL,
				(
					thread_id,
					checkpoint_ns,
					checkpoint["id"],
					checkpoint_id,
					json.dumps(copy),
					json.dumps(get_checkpoint_metadata(config, metadata)),
				),
			)
		return next_config

	async def aput_writes(
		self,
		config: RunnableConfig,
		writes: Sequence[tuple[str, Any]],
		task_id: str,
		task_path: str = "",
	) -> None:
		"""Store intermediate writes linked to a checkpoint asynchronously.

		This method saves intermediate writes associated with a checkpoint to the SingleStore database.

		Args:
		config: Configuration of the related checkpoint.
		writes: List of writes to store.
		task_id: Identifier for the task creating the writes.
		task_path: Path of the task creating the writes.
		"""
		query = (
			self.UPSERT_CHECKPOINT_WRITES_SQL
			if all(w[0] in WRITES_IDX_MAP for w in writes)
			else self.INSERT_CHECKPOINT_WRITES_SQL
		)

		write_data_list = await asyncio.to_thread(
			self._dump_writes,
			config["configurable"]["thread_id"],
			config["configurable"]["checkpoint_ns"],
			config["configurable"]["checkpoint_id"],
			task_id,
			task_path,
			writes,
		)

		async with self._cursor() as cur:
			for write_data in write_data_list:
				await asyncio.to_thread(cur.execute, query, write_data)

	async def adelete_thread(self, thread_id: str) -> None:
		"""Delete all checkpoints and writes associated with a thread ID asynchronously.

		Args:
		thread_id: The thread ID to delete.

		Returns:
		None
		"""
		async with self._cursor() as cur:
			await asyncio.to_thread(
				cur.execute,
				"DELETE FROM checkpoints WHERE thread_id = %s",
				(str(thread_id),),
			)
			await asyncio.to_thread(
				cur.execute,
				"DELETE FROM checkpoint_blobs WHERE thread_id = %s",
				(str(thread_id),),
			)
			await asyncio.to_thread(
				cur.execute,
				"DELETE FROM checkpoint_writes WHERE thread_id = %s",
				(str(thread_id),),
			)

	@asynccontextmanager
	async def _cursor(self) -> AsyncIterator[Cursor]:
		"""Create a database cursor as a context manager."""
		async with self.lock, _ainternal.get_connection(self.conn) as conn:
			with conn.cursor() as cur:
				yield cur

	async def _load_checkpoint_tuple(self, value: dict[str, Any]) -> CheckpointTuple:
		"""
		Convert a database row into a CheckpointTuple object asynchronously.

		Args:
		value: A row from the database containing checkpoint data.

		Returns:
		CheckpointTuple: A structured representation of the checkpoint,
		including its configuration, metadata, parent checkpoint (if any),
		and pending writes.
		"""

		# Parse channel_values JSON array and convert to the format expected by _load_blobs
		channel_values_raw = value.get("channel_values")
		channel_values_parsed = []
		if channel_values_raw:
			import json

			try:
				channel_array = (
					json.loads(channel_values_raw) if isinstance(channel_values_raw, str) else channel_values_raw
				)
				if channel_array:  # Only process if not empty
					channel_values_parsed = [
						(
							item[0].encode("utf-8"),  # channel as bytes
							item[1].encode("utf-8"),  # type as bytes
							bytes.fromhex(item[2]) if item[2] else b"",  # blob from hex to bytes
						)
						for item in channel_array
					]
			except (json.JSONDecodeError, IndexError, ValueError) as e:
				print(f"Error parsing channel_values: {e}")
				channel_values_parsed = []

		# Parse pending_writes JSON array and convert to the format expected by _load_writes
		pending_writes_raw = value.get("pending_writes")
		pending_writes_parsed = []
		if pending_writes_raw:
			try:
				writes_array = (
					json.loads(pending_writes_raw) if isinstance(pending_writes_raw, str) else pending_writes_raw
				)
				if writes_array:  # Only process if not empty
					pending_writes_parsed = [
						(
							item[0].encode("utf-8"),  # task_id as bytes
							item[1].encode("utf-8"),  # channel as bytes
							item[2].encode("utf-8"),  # type as bytes
							bytes.fromhex(item[3]) if item[3] else b"",  # blob from hex to bytes
						)
						for item in writes_array
					]
			except (json.JSONDecodeError, IndexError, ValueError) as e:
				print(f"Error parsing pending_writes: {e}")
				pending_writes_parsed = []

		return CheckpointTuple(
			{
				"configurable": {
					"thread_id": value["thread_id"],
					"checkpoint_ns": value["checkpoint_ns"],
					"checkpoint_id": value["checkpoint_id"],
				}
			},
			{
				**value["checkpoint"],
				"channel_values": {
					**value["checkpoint"].get("channel_values"),
					**self._load_blobs(channel_values_parsed),
				},
			},
			value["metadata"],
			(
				{
					"configurable": {
						"thread_id": value["thread_id"],
						"checkpoint_ns": value["checkpoint_ns"],
						"checkpoint_id": value["parent_checkpoint_id"],
					}
				}
				if value["parent_checkpoint_id"]
				else None
			),
			self._load_writes(pending_writes_parsed),
		)

	def list(
		self,
		config: RunnableConfig | None,
		*,
		filter: dict[str, Any] | None = None,
		before: RunnableConfig | None = None,
		limit: int | None = None,
	) -> Iterator[CheckpointTuple]:
		"""List checkpoints from the database.

		This method retrieves a list of checkpoint tuples from the SingleStore database based
		on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

		Args:
		config: Base configuration for filtering checkpoints.
		filter: Additional filtering criteria for metadata.
		before: If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
		limit: Maximum number of checkpoints to return.

		Yields:
		Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
		"""
		try:
			# check if we are in the main thread, only bg threads can block
			# we don't check in other methods to avoid the overhead
			if asyncio.get_running_loop() is self.loop:
				raise asyncio.InvalidStateError(
					"Synchronous calls to AsyncSingleStoreSaver are only allowed from a "
					"different thread. From the main thread, use the async interface. "
					"For example, use `checkpointer.alist(...)` or `await "
					"graph.ainvoke(...)`."
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
		"""Get a checkpoint tuple from the database.

		This method retrieves a checkpoint tuple from the SingleStore database based on the
		provided config. If the config contains a "checkpoint_id" key, the checkpoint with
		the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
		for the given thread ID is retrieved.

		Args:
		config: The config to use for retrieving the checkpoint.

		Returns:
		Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
		"""
		try:
			# check if we are in the main thread, only bg threads can block
			# we don't check in other methods to avoid the overhead
			if asyncio.get_running_loop() is self.loop:
				raise asyncio.InvalidStateError(
					"Synchronous calls to AsyncSingleStoreSaver are only allowed from a "
					"different thread. From the main thread, use the async interface. "
					"For example, use `await checkpointer.aget_tuple(...)` or `await "
					"graph.ainvoke(...)`."
				)
		except RuntimeError:
			pass
		return asyncio.run_coroutine_threadsafe(self.aget_tuple(config), self.loop).result()

	def get(self, config: RunnableConfig) -> Checkpoint | None:
		"""Get a checkpoint from the database.

		This method retrieves a checkpoint from the SingleStore database based on the
		provided config. If the config contains a "checkpoint_id" key, the checkpoint with
		the matching thread ID and timestamp is retrieved. Otherwise, the latest checkpoint
		for the given thread ID is retrieved.

		Args:
		config: The config to use for retrieving the checkpoint.

		Returns:
		Optional[Checkpoint]: The retrieved checkpoint, or None if no matching checkpoint was found.
		"""
		checkpoint_tuple = self.get_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	async def aget(self, config: RunnableConfig) -> Checkpoint | None:
		"""Asynchronously get a checkpoint from the database.

		Args:
		config: The config to use for retrieving the checkpoint.

		Returns:
		Optional[Checkpoint]: The retrieved checkpoint, or None if no matching checkpoint was found.
		"""
		checkpoint_tuple = await self.aget_tuple(config)
		return checkpoint_tuple.checkpoint if checkpoint_tuple else None

	def put(
		self,
		config: RunnableConfig,
		checkpoint: Checkpoint,
		metadata: CheckpointMetadata,
		new_versions: ChannelVersions,
	) -> RunnableConfig:
		"""Save a checkpoint to the database.

		This method saves a checkpoint to the SingleStore database. The checkpoint is associated
		with the provided config and its parent config (if any).

		Args:
		config: The config to associate with the checkpoint.
		checkpoint: The checkpoint to save.
		metadata: Additional metadata to save with the checkpoint.
		new_versions: New channel versions as of this write.

		Returns:
		RunnableConfig: Updated configuration after storing the checkpoint.
		"""
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
		"""Store intermediate writes linked to a checkpoint.

		This method saves intermediate writes associated with a checkpoint to the SingleStore database.

		Args:
		config: Configuration of the related checkpoint.
		writes: List of writes to store.
		task_id: Identifier for the task creating the writes.
		task_path: Path of the task creating the writes.
		"""
		return asyncio.run_coroutine_threadsafe(
			self.aput_writes(config, writes, task_id, task_path), self.loop
		).result()

	def delete_thread(self, thread_id: str) -> None:
		"""Delete all checkpoints and writes associated with a thread ID.

		Args:
		thread_id: The thread ID to delete.

		Returns:
		None
		"""
		try:
			# check if we are in the main thread, only bg threads can block
			# we don't check in other methods to avoid the overhead
			if asyncio.get_running_loop() is self.loop:
				raise asyncio.InvalidStateError(
					"Synchronous calls to AsyncSingleStoreSaver are only allowed from a "
					"different thread. From the main thread, use the async interface. "
					"For example, use `await checkpointer.adelete_thread(...)` or `await "
					"graph.ainvoke(...)`."
				)
		except RuntimeError:
			pass
		return asyncio.run_coroutine_threadsafe(self.adelete_thread(thread_id), self.loop).result()


__all__ = ["AsyncSingleStoreSaver"]
