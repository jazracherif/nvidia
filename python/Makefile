
PYENV=.venv

.PHONY: venv
venv:
	python3 -m venv ${PYENV}

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: clean
clean:
	rm -rf ${PYENV}