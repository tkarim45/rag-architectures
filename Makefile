.PHONY: install test bench
install:
	pip install -e ".[dev]"
test:
	pytest -q
bench:
	rag-bench
