default:
    just --list

# Development
sync:
    uv sync --frozen --all-groups

lock:
    uv lock

format:
    uv run --frozen ruff check --fix src tests
    uv run --frozen ruff format src tests

check:
    uv run --frozen prek run --all-files

# Testing
test:
    uv run --frozen pytest

# Release
build:
    uv build --clear
