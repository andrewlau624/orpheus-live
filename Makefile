.PHONY: install run voices lint format test pull-models clean

install:
	uv sync

# `make run` = random voice; `make run Emma` = saved preset "Emma".
run:
	uv run python -m orpheus_live $(filter-out $@,$(MAKECMDGOALS))

voices:
	uv run python -m orpheus_live --list

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

pull-models:
	ollama pull llama3.2:3b
	ollama pull llama3.2:1b

clean:
	rm -rf .pytest_cache **/__pycache__ dist build *.egg-info

# Swallow extra goals (the voice name after `run`) so make doesn't error on them.
%:
	@:
