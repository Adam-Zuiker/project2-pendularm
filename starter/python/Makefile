.PHONY: build run clean test
build:
	python3 -m py_compile src/main.py
run:
	python3 src/main.py
# Your own tests in tests/ (tests/README.md): start `make run` first.
test:
	@python3 tests/run_tests.py $(TESTS)
clean:
	rm -rf src/__pycache__
