PYTHONPATH := $(PWD):$(PYTHONPATH)

test:
	py.test --pep8 --verbose --clearcache
