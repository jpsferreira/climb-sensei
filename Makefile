.PHONY: install
install: ## Install the project using uv
	@echo "🚀 Creating virtual environment and installing dependencies with uv"
	@uv sync
	@echo "✅ Installation complete! Activate with: source .venv/bin/activate"

.PHONY: install-dev
install-dev: ## Install with development dependencies
	@echo "🚀 Installing with development dependencies"
	@uv sync --extra dev
	@uv run pre-commit install
	@echo "✅ Development installation complete!"

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "🚀 Installing pre-commit hooks"
	@uv run pre-commit install
	@echo "✅ Pre-commit hooks installed!"

.PHONY: pre-commit
pre-commit: ## Run pre-commit on all files
	@echo "🚀 Running pre-commit checks"
	@uv run pre-commit run --all-files
	@echo "✅ Pre-commit checks complete!"

.PHONY: check
check: ## Run code quality tools (ruff linting)
	@echo "🚀 Linting code: Running ruff check"
	@uv run ruff check .
	@echo "✅ All checks passed!"

.PHONY: format
format: ## Format code with ruff and black
	@echo "🚀 Formatting code: Running ruff format"
	@uv run ruff format .
	@echo "🚀 Formatting code: Running black"
	@uv run black .
	@echo "✅ Code formatted!"

.PHONY: lint-fix
lint-fix: ## Fix linting issues automatically
	@echo "🚀 Fixing linting issues: Running ruff check --fix"
	@uv run ruff check --fix .
	@echo "✅ Linting fixes applied!"

.PHONY: test
test: ## Run tests with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run pytest --cov --cov-config=pyproject.toml --cov-report=term-missing
	@echo "✅ Tests complete!"

.PHONY: test-fast
test-fast: ## Run tests without coverage
	@echo "🚀 Running fast tests (no coverage)"
	@uv run pytest -v
	@echo "✅ Tests complete!"

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "🚀 Running tests in watch mode"
	@uv run pytest-watch

.PHONY: coverage
coverage: ## Generate HTML coverage report
	@echo "🚀 Generating coverage report"
	@uv run pytest --cov --cov-config=pyproject.toml --cov-report=html
	@echo "✅ Coverage report generated in htmlcov/index.html"
	@open htmlcov/index.html || xdg-open htmlcov/index.html

.PHONY: run
run: ## Run the interactive demo
	@echo "🚀 Running climb-sensei demo"
	@python -m climb_sensei

.PHONY: analyze
analyze: ## Analyze a video (usage: make analyze VIDEO=path/to/video.mp4)
	@echo "🚀 Analyzing video: $(VIDEO)"
	@python scripts/analyze_climb.py $(VIDEO) --video output.mp4 --json analysis.json

.PHONY: clean
clean: ## Clean build artifacts, cache, and temporary files
	@echo "🧹 Cleaning up..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf .eggs/
	@rm -rf *.egg-info
	@rm -rf *.egg
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .ruff_cache
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.pyc' -delete
	@find . -type f -name '*.pyo' -delete
	@find . -type f -name '*~' -delete
	@echo "✅ Cleanup complete!"

.PHONY: clean-test
clean-test: ## Remove test and coverage artifacts
	@echo "🧹 Cleaning test artifacts..."
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf .tox/
	@echo "✅ Test artifacts removed!"

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Building wheel file"
	@uv build
	@echo "✅ Build complete!"

.PHONY: docs
docs: ## Serve documentation locally
	@echo "🚀 Serving documentation at http://127.0.0.1:8000"
	@uv run mkdocs serve

.PHONY: docs-build
docs-build: ## Build documentation site
	@echo "🚀 Building documentation"
	@uv run mkdocs build
	@echo "✅ Documentation built in site/"

.PHONY: docs-deploy
docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "🚀 Deploying documentation to GitHub Pages"
	@uv run mkdocs gh-deploy --force
	@echo "✅ Documentation deployed!"

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf dist/
	@rm -rf build/
	@rm -rf *.egg-info
	@echo "✅ Build artifacts removed!"

.PHONY: all
all: clean install-dev check test ## Run all checks and tests
	@echo "✅ All tasks complete!"

.PHONY: qa
qa: format lint-fix test ## Run full QA pipeline (format, lint, test)
	@echo "✅ QA pipeline complete!"

.PHONY: update
update: ## Update dependencies
	@echo "🚀 Updating dependencies"
	@uv sync --upgrade
	@echo "✅ Dependencies updated!"

.PHONY: lock
lock: ## Update lock file
	@echo "🚀 Updating lock file"
	@uv lock
	@echo "✅ Lock file updated!"

.PHONY: help
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
