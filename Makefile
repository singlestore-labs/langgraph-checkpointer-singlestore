.PHONY: lint-check lint-fix format-check format-fix type-check run-checks deps install tests uninstall clean release setup-db teardown-db reset-db status-db test-connection

lint-check:
	uv run ruff check

lint-fix:
	uv run ruff check --fix

format-check:
	uv run ruff format --check

format-fix:
	uv run ruff format

type-check:
	uv run pyrefly check

run-checks: lint-check format-check

deps:
	uv sync --locked --all-extras --dev

install: deps
	uv pip install -e .

tests:
	uv run pytest

test-http-live: ## Run HTTP tests with specific server URL
	@echo "Usage: make test-http-live SERVER_URL=http://localhost:8080 BASE_PATH=/api/v1 API_KEY=your-key"
	@if [ -z "$(SERVER_URL)" ]; then echo "ERROR: SERVER_URL is required"; exit 1; fi
	uv run pytest tests/test_real_server.py \
		--use-real-server \
		--server-url="$(SERVER_URL)" \
		$(if $(BASE_PATH),--base-path="$(BASE_PATH)") \
		$(if $(API_KEY),--api-key="$(API_KEY)") \
		-xvs

uninstall:
	uv pip uninstall .

clean: ## Clean up generated files and environment
	uv pip uninstall .
	uv clean
	rm -rf .venv .ruff_cache .pytest_cache
	rm -rf **/*/*.egg-info **/*/__pycache__
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ __pycache__/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

release:
	uv run scripts/release.py

setup-db: ## Start SingleStore database
	@echo "🚀 Starting SingleStore database..."
	@docker compose -f tests/compose-singlestore.yml down -v >/dev/null 2>&1 || true
	@docker compose -f tests/compose-singlestore.yml up -d >/dev/null 2>&1
	@echo "⏳ Waiting for database to be ready..."
	@timeout 30 bash -c 'i=0; until docker compose -f tests/compose-singlestore.yml ps | grep -q "healthy"; do i=$$((i+3)); echo "⏳ Still waiting for database... ($$i/30s)"; sleep 3; done; echo "✅ Database is ready!"' || (echo "❌ Database failed to start within 30 seconds"; exit 1)
	@echo "📝 Initializing database schema..."
	@docker compose -f tests/compose-singlestore.yml exec -T singlestore-test singlestore -u root -ptest < tests/init.sql >/dev/null 2>&1 && echo "✅ Schema initialized successfully"
	@echo "✅ Database ready and initialized"
	@echo "   • Port: 33071"
	@echo "   • Web UI: http://localhost:18091"
	@echo "   • Test databases:"
	@docker compose -f tests/compose-singlestore.yml exec -T singlestore-test singlestore -u root -ptest -e "SHOW DATABASES;" 2>/dev/null | grep -E "test" | sed 's/^/     /'

teardown-db: ## Stop SingleStore database
	@echo "🛑 Stopping SingleStore database..."
	@docker compose -f tests/compose-singlestore.yml down -v >/dev/null 2>&1
	@echo "✅ Database stopped"

reset-db: teardown-db setup-db ## Reset SingleStore database (stop and start fresh)

status-db: ## Check SingleStore database status
	@echo "🔍 Checking database status..."
	@if docker compose -f tests/compose-singlestore.yml ps | grep -q "singlestore-test.*Up"; then \
		echo "✅ SingleStore database is running"; \
		echo "   • Port: 33071"; \
		echo "   • Web UI: http://localhost:18091"; \
		echo "   • Management: http://localhost:19191"; \
	else \
		echo "❌ SingleStore database is not running"; \
		echo "   Run 'make setup-db' to start it"; \
	fi

test-connection: ## Test connection from host using custom port
	@echo "🔍 Testing database connection..."
	@echo "⏳ Attempting to connect..."
	@if docker compose -f tests/compose-singlestore.yml exec -T singlestore-test singlestore -u root -ptest -e "SHOW DATABASES;" >/dev/null 2>&1; then \
		echo "✅ Connection successful - database is responding"; \
	else \
		echo "❌ Connection failed - ensure SingleStore is running"; \
		echo "   Try: make setup-db"; \
	fi