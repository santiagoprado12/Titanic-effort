VENV=venv
virtual_env:
	python3 -m pip install virtualenv
	python3 -m virtualenv $(VENV)

install: virtual_env
	. $(VENV)/bin/activate && python3 -m pip install --upgrade pip
	. $(VENV)/bin/activate && python3 -m pip install -r requirements.txt

