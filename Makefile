PYTHON = uv run python
FLAKE8 = uv run flake8
MYPY = uv run mypy
SRC_DIR = src
LLM_SDK_DIR = llm_sdk
all: run
install:
		uv sync
run:
		$(PYTHON) -m $(SRC_DIR)
debug:
		$(PYTHON) -m pdb $(SRC_DIR)
clean:
		rm -rf .venv
		find . -type d -name "__pycache__" -exec rm -rf {} +
		find . -type d -name ".mypy_cache" -exec rm -rf {} +
		find . -type d -name ".pytest_cache" -exec rm -rf {} +
lint:
		$(FLAKE8) $(SRC_DIR)
		$(MYPY) $(SRC_DIR)

lint-strict:
		$(FLAKE8) $(SRC_DIR)
	$(MYPY) --strict $(SRC_DIR)

.PHONY: all install run debug clean lint lint-stricT