default:
    just --list

# Development
sync:
    uv sync --frozen --all-groups

lock:
    uv lock

format:
    uv run --frozen ruff check --fix src tests benchmarks
    uv run --frozen ruff format src tests benchmarks

check:
    uv run --frozen prek run --all-files

# Testing
test:
    uv run --frozen pytest

test-and-report:
    uv run --frozen pytest --cov=src/jaxellip --cov-report=term-missing --cov-report=html

coverage-report:
    uv run --frozen coverage combine coverage
    uv run --frozen coverage report
    uv run --frozen coverage html --show-contexts --title "jaxellip coverage"

# Benchmarking
benchmark:
    uv run --frozen python benchmarks/compare_to_scipy.py

# Release
build:
    uv build --clear
