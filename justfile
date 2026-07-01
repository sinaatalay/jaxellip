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

benchmark:
    uv run --frozen python benchmarks/compare_scipy.py

# Release
build:
    uv build --clear
