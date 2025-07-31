.PHONY: lint-check lint-fix format-check format-fix type-check run-checks deps install tests uninstall clean release setup-db teardown-db reset-db test-connection

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

run-checks: lint-check format-check type-check

deps:
	uv sync --locked --all-extras --dev

install: deps
	uv pip install -e .

tests:
	uv run pytest

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
	@echo "Starting SingleStore database..."
	@echo "Checking for existing containers..."
	@docker compose -f tests/compose-singlestore.yml down -v 2>/dev/null || true
	@echo "Starting fresh container..."
	docker compose -f tests/compose-singlestore.yml up -d
	@echo "Waiting for database to be ready and initialized..."
	@echo "Waiting for database..."
	@until docker compose -f tests/compose-singlestore.yml exec -T singlestore-test singlestore -u root -ptest_password_123 -e "SELECT 1" >/dev/null 2>&1; do echo "Waiting for database..."; sleep 5; done
	@echo "Creating test databases..."
	@docker compose -f tests/compose-singlestore.yml exec -T singlestore-test singlestore -u root -ptest_password_123 -e "CREATE DATABASE IF NOT EXISTS test_db; CREATE DATABASE IF NOT EXISTS test_example; CREATE DATABASE IF NOT EXISTS test_example_async;"
	@echo "Verifying database initialization..."
	@docker compose -f tests/compose-singlestore.yml exec -T singlestore-test singlestore -u root -ptest_password_123 -e "SHOW DATABASES;" | grep test_db || echo "Database initialization pending..."

teardown-db: ## Stop SingleStore database
	@echo "Stopping SingleStore database..."
	docker compose -f tests/compose-singlestore.yml down -v

reset-db: teardown-db setup-db ## Reset SingleStore database (stop and start fresh)

test-connection: ## Test connection from host using custom port
	mysql -h localhost -P 33071 -u root -ptest_password_123 -e "SHOW DATABASES;" 2>/dev/null || echo "❌ Cannot connect from host - ensure SingleStore is running"