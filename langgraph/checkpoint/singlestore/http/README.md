# HTTP-Based SingleStore Checkpointer

An HTTP-based implementation of the SingleStore checkpointer that communicates with a remote checkpoint server via REST APIs instead of direct database connections.

## Features

- **No Direct DB Access**: Checkpointer communicates entirely through HTTP APIs
- **Connection Pooling**: Efficient HTTP connection management using httpx
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
- **Async Support**: Both sync and async implementations available
- **Binary Data Handling**: Base64 encoding for blob/binary checkpoint data
- **Full Compatibility**: Drop-in replacement for direct SingleStore checkpointer

## Installation

```bash
pip install httpx
```

## Usage

### Synchronous Client

```python
from langgraph.checkpoint.singlestore.http import HTTPSingleStoreSaver

# Basic usage
saver = HTTPSingleStoreSaver(
    base_url="http://checkpoint-server:8080",
    api_key="your-api-key"
)

# With custom configuration
from langgraph.checkpoint.singlestore.http import RetryConfig

saver = HTTPSingleStoreSaver(
    base_url="http://checkpoint-server:8080",
    api_key="your-api-key",
    timeout=60.0,  # Request timeout in seconds
    retry_config=RetryConfig(
        max_retries=5,
        backoff_factor=0.2,
        retry_statuses={429, 500, 502, 503, 504}
    ),
    pool_connections=20,  # Connection pool size
    pool_maxsize=30       # Max pool size
)

# Using context manager
with HTTPSingleStoreSaver.from_url(
    base_url="http://checkpoint-server:8080",
    api_key="your-api-key"
) as saver:
    # Setup database tables (first time only)
    saver.setup()
    
    # Use with LangGraph
    from langgraph.graph import Graph
    
    graph = Graph()
    # ... build your graph ...
    
    # Compile with checkpointer
    app = graph.compile(checkpointer=saver)
```

### Asynchronous Client

```python
from langgraph.checkpoint.singlestore.http.aio import AsyncHTTPSingleStoreSaver

async def main():
    # Create async saver
    async with AsyncHTTPSingleStoreSaver.from_url(
        base_url="http://checkpoint-server:8080",
        api_key="your-api-key",
        max_connections=100,
        max_keepalive_connections=30
    ) as saver:
        # Setup database
        await saver.setup()
        
        # Use with async LangGraph
        from langgraph.graph import Graph
        
        graph = Graph()
        # ... build your graph ...
        
        app = graph.compile(checkpointer=saver)
        
        # Run async
        result = await app.ainvoke(...)
```

## API Endpoints Required

The HTTP server must implement these endpoints:

### Setup & Migration
- `POST /setup` - Initialize database tables and run migrations
  
### Checkpoint Operations
- `GET /checkpoints` - List checkpoints with filtering
- `GET /checkpoints/{thread_id}/{checkpoint_ns}/{checkpoint_id}` - Get specific checkpoint
- `GET /checkpoints/{thread_id}/{checkpoint_ns}/latest` - Get latest checkpoint
- `PUT /checkpoints` - Create/update checkpoint with blob data
- `DELETE /threads/{thread_id}` - Delete all data for a thread

### Checkpoint Writes
- `PUT /checkpoint-writes` - Store intermediate checkpoint writes

## Configuration Options

### HTTPSingleStoreSaver

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | required | Base URL of checkpoint server |
| `api_key` | str | None | API key for authentication |
| `serde` | SerializerProtocol | None | Custom serializer |
| `timeout` | float | 30.0 | Request timeout in seconds |
| `retry_config` | RetryConfig | Default | Retry configuration |
| `pool_connections` | int | 10 | Connection pool size |
| `pool_maxsize` | int | 20 | Maximum pool size |

### RetryConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | int | 3 | Maximum retry attempts |
| `backoff_factor` | float | 0.1 | Exponential backoff multiplier |
| `retry_statuses` | set[int] | {429, 500, 502, 503, 504} | HTTP status codes to retry |

## Performance Optimizations

1. **Connection Pooling**: Reuses HTTP connections across requests
2. **HTTP/2 Support**: Enabled by default for better multiplexing
3. **Request Batching**: Blob data sent in batches when possible
4. **Async Operations**: Non-blocking I/O for high concurrency
5. **Retry Logic**: Automatic handling of transient failures

## Error Handling

```python
from langgraph.checkpoint.singlestore.http import HTTPClientError

try:
    checkpoint = saver.get(config)
except HTTPClientError as e:
    print(f"HTTP Error {e.status_code}: {e.message}")
    if e.details:
        print(f"Details: {e.details}")
```

## Testing

The implementation includes comprehensive tests using pytest-httpx:

```bash
# Install test dependencies
pip install pytest pytest-httpx pytest-asyncio

# Run tests
pytest tests/test_http_sync.py
pytest tests/test_http_async.py
```

## Migration from Direct SingleStore

Migrating from direct database connection to HTTP-based:

```python
# Before (direct connection)
from langgraph.checkpoint.singlestore import SingleStoreSaver
saver = SingleStoreSaver(conn)

# After (HTTP-based)
from langgraph.checkpoint.singlestore.http import HTTPSingleStoreSaver
saver = HTTPSingleStoreSaver(
    base_url="http://checkpoint-server:8080",
    api_key="your-api-key"
)
```

## Security Considerations

1. **Authentication**: Always use API keys or bearer tokens
2. **HTTPS**: Use HTTPS in production for encrypted communication
3. **Rate Limiting**: Server should implement rate limiting
4. **Input Validation**: Server must validate all inputs
5. **Access Control**: Implement proper authorization on server

## Example Server Implementation

A minimal Flask server example:

```python
from flask import Flask, request, jsonify
import singlestoredb

app = Flask(__name__)

@app.route('/setup', methods=['POST'])
def setup():
    # Run database migrations
    # ... migration logic ...
    return jsonify({"success": True, "version": 10})

@app.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    # Query checkpoints from database
    # ... query logic ...
    return jsonify({"checkpoints": [], "total": 0})

@app.route('/checkpoints', methods=['PUT'])
def put_checkpoint():
    data = request.json
    # Store checkpoint in database
    # ... storage logic ...
    return jsonify({})

# ... implement other endpoints ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Benefits

- **Decoupling**: Application doesn't need database drivers
- **Scalability**: HTTP server can be load-balanced
- **Security**: Centralized authentication and authorization
- **Flexibility**: Server can use any database backend
- **Monitoring**: Standard HTTP metrics and logging
- **Caching**: Can leverage HTTP caching mechanisms