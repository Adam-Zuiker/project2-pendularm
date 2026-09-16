.PHONY: build run clean
build:
	python3 -m compileall src
run:
	python3 src/main.py
clean:
	find src -type d -name __pycache__ -exec rm -rf {} +
