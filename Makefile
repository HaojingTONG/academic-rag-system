# Academic RAG System - Makefile
# =================================

.PHONY: help install dev test lint format clean run index serve docker smoke

# Configuration
PYTHON := python3
VENV := venv_m3max
BIN := $(VENV)/bin
PROJECT := academic-rag-system

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help:  ## Show this help message
	@echo "$(BLUE)$(PROJECT) - Available Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# Installation & Setup
# ---------------------

install:  ## Install production dependencies
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(RESET)"

dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PYTHON) -m pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development setup complete$(RESET)"

venv:  ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Virtual environment created$(RESET)"
	@echo "Activate with: source $(VENV)/bin/activate"

bootstrap:  ## Run full development setup
	@echo "$(BLUE)Running bootstrap script...$(RESET)"
	bash scripts/dev_bootstrap.sh

# Testing
# --------

test:  ## Run all tests with coverage
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(BIN)/pytest tests/ -v --cov=rag --cov=indexer --cov=models --cov-report=html --cov-report=term-missing

test-unit:  ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(BIN)/pytest tests/unit/ -v

test-integration:  ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(BIN)/pytest tests/integration/ -v -m integration

test-regression:  ## Run regression tests
	@echo "$(BLUE)Running regression tests...$(RESET)"
	$(BIN)/pytest tests/regression/ -v -m regression

test-fast:  ## Run fast tests only (skip slow ones)
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(BIN)/pytest tests/ -v -m "not slow"

smoke:  ## Run smoke tests
	@echo "$(BLUE)Running smoke tests...$(RESET)"
	bash scripts/run_smoke.sh

# Code Quality
# ------------

lint:  ## Run linters (ruff, mypy)
	@echo "$(BLUE)Running linters...$(RESET)"
	$(BIN)/ruff check rag/ indexer/ models/ app/ tests/
	$(BIN)/mypy rag/ indexer/ models/ app/

lint-fix:  ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(RESET)"
	$(BIN)/ruff check --fix rag/ indexer/ models/ app/ tests/

format:  ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(BIN)/black rag/ indexer/ models/ app/ tests/
	$(BIN)/ruff check --fix rag/ indexer/ models/ app/ tests/
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format-check:  ## Check code formatting without changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(BIN)/black --check rag/ indexer/ models/ app/ tests/

# Cleaning
# --------

clean:  ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"

clean-data:  ## Clean cached data (embedding cache, logs)
	@echo "$(YELLOW)Warning: This will delete cached embeddings and logs$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/embedding_cache/*; \
		rm -rf logs/*; \
		echo "$(GREEN)✓ Data cleaned$(RESET)"; \
	fi

clean-all: clean clean-data  ## Clean everything including data

# Data & Indexing
# ---------------

index:  ## Build vector index from PDFs
	@echo "$(BLUE)Building vector index...$(RESET)"
	$(BIN)/python scripts/process_pdf_fulltext.py
	@echo "$(GREEN)✓ Index built$(RESET)"

index-quick:  ## Quick index (process only new PDFs)
	@echo "$(BLUE)Building quick index...$(RESET)"
	$(BIN)/python scripts/process_pdf_fulltext.py --incremental
	@echo "$(GREEN)✓ Quick index built$(RESET)"

download-papers:  ## Download papers from arXiv
	@echo "$(BLUE)Downloading papers...$(RESET)"
	$(BIN)/python scripts/download_pdfs.py
	@echo "$(GREEN)✓ Papers downloaded$(RESET)"

collect-papers:  ## Collect classic papers metadata
	@echo "$(BLUE)Collecting papers metadata...$(RESET)"
	$(BIN)/python scripts/collect_classic_papers.py
	@echo "$(GREEN)✓ Papers collected$(RESET)"

# Running
# -------

run:  ## Run CLI in interactive mode
	@echo "$(BLUE)Starting RAG system CLI...$(RESET)"
	$(BIN)/python -m app.cli query --interactive

query:  ## Run single query (usage: make query Q="your question")
	@if [ -z "$(Q)" ]; then \
		echo "$(RED)Error: Please provide a query with Q=\"your question\"$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Querying: $(Q)$(RESET)"
	$(BIN)/python -m app.cli query "$(Q)"

serve:  ## Start FastAPI server
	@echo "$(BLUE)Starting FastAPI server...$(RESET)"
	$(BIN)/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:  ## Start FastAPI in production mode
	@echo "$(BLUE)Starting FastAPI (production)...$(RESET)"
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Evaluation
# ----------

evaluate:  ## Run RAG system evaluation
	@echo "$(BLUE)Running evaluation...$(RESET)"
	$(BIN)/python scripts/evaluate_rag_system.py

evaluate-baseline:  ## Create baseline metrics
	@echo "$(BLUE)Creating baseline metrics...$(RESET)"
	$(BIN)/python scripts/evaluate_rag_system.py --save-baseline

benchmark:  ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(RESET)"
	$(BIN)/python scripts/benchmark.py

# Docker
# ------

docker-build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(PROJECT):latest .
	@echo "$(GREEN)✓ Docker image built$(RESET)"

docker-run:  ## Run in Docker
	@echo "$(BLUE)Starting Docker container...$(RESET)"
	docker run -p 8000:8000 -v $(PWD)/data:/app/data $(PROJECT):latest

docker-compose-up:  ## Start with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(RESET)"
	docker-compose up -d

docker-compose-down:  ## Stop docker-compose services
	@echo "$(BLUE)Stopping services...$(RESET)"
	docker-compose down

# Documentation
# -------------

docs:  ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	$(BIN)/sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)✓ Documentation generated$(RESET)"

docs-serve:  ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(RESET)"
	$(BIN)/python -m http.server 8080 -d docs/_build/html

# Git & Version Control
# ---------------------

commit:  ## Interactive commit helper
	@echo "$(BLUE)Running pre-commit checks...$(RESET)"
	$(BIN)/pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks passed$(RESET)"
	git add -A
	git status
	@echo "$(YELLOW)Ready to commit. Run: git commit -m 'your message'$(RESET)"

tag:  ## Create git tag (usage: make tag VERSION=v2.0.0)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Error: Please provide VERSION=vX.Y.Z$(RESET)"; \
		exit 1; \
	fi
	git tag -a $(VERSION) -m "Release $(VERSION)"
	@echo "$(GREEN)✓ Tag $(VERSION) created$(RESET)"
	@echo "Push with: git push origin $(VERSION)"

# Utility
# -------

status:  ## Show system status
	@echo "$(BLUE)System Status:$(RESET)"
	@echo ""
	@echo "Python Version:"
	@$(PYTHON) --version
	@echo ""
	@echo "Ollama Status:"
	@curl -s http://localhost:11434/api/version | jq . || echo "$(RED)Ollama not running$(RESET)"
	@echo ""
	@echo "Vector DB Size:"
	@du -sh vector_db 2>/dev/null || echo "No vector DB found"
	@echo ""
	@echo "Data Directory:"
	@du -sh data 2>/dev/null || echo "No data directory"

check-deps:  ## Check if required services are running
	@echo "$(BLUE)Checking dependencies...$(RESET)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)✗ Python not found$(RESET)"; exit 1; }
	@echo "$(GREEN)✓ Python found$(RESET)"
	@curl -s http://localhost:11434/api/version >/dev/null 2>&1 && echo "$(GREEN)✓ Ollama running$(RESET)" || echo "$(YELLOW)⚠ Ollama not running$(RESET)"

info:  ## Show project information
	@echo "$(BLUE)$(PROJECT)$(RESET)"
	@echo "Version: 2.0.0"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Virtual Env: $(VENV)"
	@echo ""
	@echo "Directory Structure:"
	@tree -L 2 -I '__pycache__|*.pyc|venv*|node_modules' || ls -la

# Development Workflow
# --------------------

dev-setup: venv install dev  ## Complete development setup
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate environment: source $(VENV)/bin/activate"
	@echo "  2. Copy .env.example to .env"
	@echo "  3. Run smoke tests: make smoke"
	@echo "  4. Build index: make index"
	@echo "  5. Run CLI: make run"

reset-db:  ## Reset vector database
	@echo "$(YELLOW)Warning: This will delete the vector database$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf vector_db/*; \
		echo "$(GREEN)✓ Vector DB reset$(RESET)"; \
		echo "Run 'make index' to rebuild"; \
	fi

fresh-start: clean-all reset-db  ## Fresh start (clean everything and rebuild)
	@echo "$(BLUE)Fresh start initiated...$(RESET)"
	$(MAKE) index
	$(MAKE) test
	@echo "$(GREEN)✓ Fresh start complete$(RESET)"

# CI/CD
# -----

ci:  ## Run CI pipeline locally
	@echo "$(BLUE)Running CI pipeline...$(RESET)"
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) test
	@echo "$(GREEN)✓ CI pipeline passed$(RESET)"

pre-push:  ## Run checks before pushing
	@echo "$(BLUE)Running pre-push checks...$(RESET)"
	$(MAKE) lint
	$(MAKE) test-fast
	@echo "$(GREEN)✓ Ready to push$(RESET)"

# Default target
.DEFAULT_GOAL := help
