PYTHONPATH := $(PWD):$(PYTHONPATH)

lint:
	flake8 .

test: lint
