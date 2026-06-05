set shell := ["bash", "-euo", "pipefail", "-c"]

setup:
	@echo "Python dependencies are provided by the declarative workstation; no repo-local pip install step is required."
	@echo "PYTHONPATH=src python -m repr_lab.main --version"
	PYTHONPATH=src python -m repr_lab.main --version

lint:
	@echo "python -m ruff check ."
	python -m ruff check .

typecheck:
	@echo "python -m mypy src"
	python -m mypy src

test:
	@echo "python -m pytest -q"
	python -m pytest -q

smoke:
	@echo "python -m ruff check ."
	python -m ruff check .
	@echo "python -m mypy src"
	python -m mypy src
	@echo "python -m pytest -q"
	python -m pytest -q
