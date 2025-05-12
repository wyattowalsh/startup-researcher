# startup reseacher Makefile

.PHONY: help setup lint format test all clean run-data-agent

# help
help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  activate: Activate the virtual environment"
	@echo "  lint: Run linting"
	@echo "  format: Run formatting"
	@echo "  test: Run tests"
	@echo "  all: Run linting, formatting, and tests"
	@echo "  run-data-agent: Run the Chainlit data agent interface"
	@echo "  clean: Clean up"

# activate
activate:
	uv venv && source .venv/bin/activate

# lint
lint:
	uv sync --group lint
	uv run -- mypy startup_researcher tests
	uv run -- pylint startup_researcher tests
	uv run -- ruff check startup_researcher tests

# format
format:
	uv sync --group format
	uv run -- isort startup_researcher tests
	uv run -- autoflake --in-place --recursive startup_researcher tests
	uv run -- autopep8 --in-place --recursive startup_researcher tests
	uv run -- black startup_researcher tests

# test
test:
	uv sync --group test
	uv run -- pytest tests/

# run-data-agent
run-data-agent:
	uv sync
	uv run -- chainlit run startup_researcher/agents/data_agent.py -w

# all
all:
	make lint
	make format
	make test

# clean
clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf logs
	rm -rf .hypothesis