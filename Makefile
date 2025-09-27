.PHONY: help lint format type-check test clean install-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt

lint: ## Run all linters
	@echo "Running flake8..."
	flake8 service/ tests/
	@echo "Running black check..."
	black --check service/ tests/
	@echo "Running isort check..."
	isort --check-only service/ tests/

format: ## Format code with black and isort
	@echo "Running black..."
	black service/ tests/
	@echo "Running isort..."
	isort service/ tests/

type-check: ## Run mypy type checking
	mypy service/

test: ## Run tests
	pytest tests/ -v

clean: ## Clean up cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

pre-commit: format lint type-check test ## Run all checks (format, lint, type-check, test)

ci: lint type-check test ## Run CI pipeline checks
