"""HTTP client with connection pooling and retry logic for SingleStore checkpointer."""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx

from .schemas import ErrorResponse


class HTTPClientError(Exception):
	"""Base exception for HTTP client errors."""

	def __init__(
		self,
		message: str,
		status_code: int | None = None,
		error_code: str | None = None,
		details: dict[str, Any] | None = None,
	):
		super().__init__(message)
		self.status_code = status_code
		self.error_code = error_code
		self.details = details


class RetryConfig:
	"""Configuration for HTTP request retry logic."""

	def __init__(
		self,
		max_retries: int = 3,
		backoff_factor: float = 0.1,
		retry_statuses: set[int] | None = None,
	):
		self.max_retries = max_retries
		self.backoff_factor = backoff_factor
		self.retry_statuses = retry_statuses or {429, 500, 502, 503, 504}


class BaseHTTPClient:
	"""Base HTTP client with shared configuration."""

	def __init__(
		self,
		base_url: str,
		base_path: str = "",
		api_key: str | None = None,
		timeout: float = 30.0,
		retry_config: RetryConfig | None = None,
		headers: dict[str, str] | None = None,
	):
		self.base_url = base_url.rstrip("/")
		self.base_path = base_path
		self.api_key = api_key
		self.timeout = httpx.Timeout(timeout=timeout, connect=10.0, pool=5.0)
		self.retry_config = retry_config or RetryConfig()
		self.headers = headers or {}

		if self.api_key:
			self.headers["Authorization"] = f"Bearer {self.api_key}"
		self.headers["Content-Type"] = "application/json"
		self.headers["Accept"] = "application/json"

	def _should_retry(self, response: httpx.Response | None, error: Exception | None) -> bool:
		"""Determine if request should be retried."""
		if error:
			# Retry on connection errors
			return bool(isinstance(error, httpx.ConnectError | httpx.ConnectTimeout | httpx.ReadTimeout))

		if response:
			return response.status_code in self.retry_config.retry_statuses

		return False

	def _calculate_backoff(self, attempt: int) -> float:
		"""Calculate exponential backoff delay."""
		return self.retry_config.backoff_factor * (2**attempt)

	def _encode_binary(self, data: bytes | None) -> str | None:
		"""Encode binary data to base64 string."""
		if data is None:
			return None
		return base64.b64encode(data).decode("utf-8")

	def _decode_binary(self, data: str | None) -> bytes | None:
		"""Decode base64 string to binary data."""
		if data is None:
			return None
		return base64.b64decode(data)

	def _handle_error_response(self, response: httpx.Response) -> None:
		"""Handle HTTP error responses."""
		if response.status_code >= 400:
			try:
				error_data = ErrorResponse.model_validate(response.json())
				raise HTTPClientError(
					message=error_data.error.message,
					details=error_data.error.details,
					error_code=error_data.error.code,
					status_code=response.status_code,
				)
			except (ValueError, KeyError) as err:
				raise HTTPClientError(
					message=f"HTTP {response.status_code}: {response.text}",
					status_code=response.status_code,
				) from err


class HTTPClient(BaseHTTPClient):
	"""Synchronous HTTP client with connection pooling and retry logic."""

	def __init__(
		self,
		base_url: str,
		base_path: str = "",
		api_key: str | None = None,
		timeout: float = 30.0,
		retry_config: RetryConfig | None = None,
		headers: dict[str, str] | None = None,
		pool_connections: int = 10,
		pool_maxsize: int = 20,
	):
		super().__init__(base_url, base_path, api_key, timeout, retry_config, headers)

		# Configure connection pooling
		self.limits = httpx.Limits(
			max_keepalive_connections=pool_connections,
			max_connections=pool_maxsize,
			keepalive_expiry=5.0,
		)

		self.client: httpx.Client | None = None

	@contextmanager
	def create(self) -> Iterator[HTTPClient]:
		"""Create client as context manager."""
		try:
			self.client = httpx.Client(
				base_url=self.base_url,
				headers=self.headers,
				timeout=self.timeout,
				limits=self.limits,
				follow_redirects=False,
				http2=True,  # Enable HTTP/2 for better performance
			)
			yield self
		finally:
			if self.client:
				self.client.close()
				self.client = None

	def _request_with_retry(
		self,
		method: str,
		path: str,
		payload: dict[str, Any] | None = None,
		params: dict[str, Any] | None = None,
	) -> httpx.Response:
		"""Execute HTTP request with retry logic."""
		if not self.client:
			raise RuntimeError("Client not initialized. Use 'with HTTPClient.create()' context manager.")

		url = f"{self.base_url}{self.base_path}{path}"
		last_error: Exception | None = None

		for attempt in range(self.retry_config.max_retries + 1):
			try:
				response = self.client.request(
					method=method,
					url=url,
					json=payload,
					params=params,
				)

				if not self._should_retry(response, None):
					self._handle_error_response(response)
					return response

			except Exception as e:
				last_error = e
				if not self._should_retry(None, e):
					raise

			# Calculate backoff delay
			if attempt < self.retry_config.max_retries:
				delay = self._calculate_backoff(attempt)
				time.sleep(delay)

		# Max retries exceeded
		if last_error:
			raise last_error
		raise HTTPClientError(f"Max retries exceeded for {method} {url}")

	def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Execute GET request."""
		response = self._request_with_retry("GET", path, params=params)
		return response.json()

	def post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
		"""Execute POST request."""
		response = self._request_with_retry("POST", path, payload=json)
		return response.json()

	def put(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
		"""Execute PUT request."""
		response = self._request_with_retry("PUT", path, payload=json)
		return response.json()

	def delete(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Execute DELETE request."""
		response = self._request_with_retry("DELETE", path, params=params)
		return response.json()


class AsyncHTTPClient(BaseHTTPClient):
	"""Asynchronous HTTP client with connection pooling and retry logic."""

	def __init__(
		self,
		base_url: str,
		base_path: str = "",
		api_key: str | None = None,
		timeout: float = 30.0,
		retry_config: RetryConfig | None = None,
		headers: dict[str, str] | None = None,
		max_connections: int = 100,
		max_keepalive_connections: int = 20,
	):
		super().__init__(base_url, base_path, api_key, timeout, retry_config, headers)

		# Configure connection pooling
		self.limits = httpx.Limits(
			max_keepalive_connections=max_keepalive_connections,
			max_connections=max_connections,
			keepalive_expiry=5.0,
		)

		self.client: httpx.AsyncClient | None = None

	@asynccontextmanager
	async def create(self) -> AsyncIterator[AsyncHTTPClient]:
		"""Create async client as context manager."""
		try:
			self.client = httpx.AsyncClient(
				base_url=self.base_url,
				headers=self.headers,
				timeout=self.timeout,
				limits=self.limits,
				follow_redirects=False,
				http2=True,  # Enable HTTP/2 for better performance
			)
			yield self
		finally:
			if self.client:
				await self.client.aclose()
				self.client = None

	async def _request_with_retry(
		self,
		method: str,
		path: str,
		json: dict[str, Any] | None = None,
		params: dict[str, Any] | None = None,
	) -> httpx.Response:
		"""Execute HTTP request with retry logic."""
		if not self.client:
			raise RuntimeError("Client not initialized. Use 'async with AsyncHTTPClient.create()' context manager.")

		url = f"{self.base_url}{self.base_path}{path}"
		last_error: Exception | None = None

		for attempt in range(self.retry_config.max_retries + 1):
			try:
				response = await self.client.request(
					method=method,
					url=url,
					json=json,
					params=params,
				)

				if not self._should_retry(response, None):
					self._handle_error_response(response)
					return response

			except Exception as e:
				last_error = e
				if not self._should_retry(None, e):
					raise

			# Calculate backoff delay
			if attempt < self.retry_config.max_retries:
				delay = self._calculate_backoff(attempt)
				await asyncio.sleep(delay)

		# Max retries exceeded
		if last_error:
			raise last_error
		raise HTTPClientError(f"Max retries exceeded for {method} {url}")

	async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Execute GET request."""
		response = await self._request_with_retry("GET", path, params=params)
		return response.json()

	async def post(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
		"""Execute POST request."""
		response = await self._request_with_retry("POST", path, json=json)
		return response.json()

	async def put(self, path: str, json: dict[str, Any]) -> dict[str, Any]:
		"""Execute PUT request."""
		response = await self._request_with_retry("PUT", path, json=json)
		return response.json()

	async def delete(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Execute DELETE request."""
		response = await self._request_with_retry("DELETE", path, params=params)
		return response.json()
