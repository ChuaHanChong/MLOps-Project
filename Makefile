##############
# ENV (NORMAL)
##############
VENV_NAME ?= env
VENV_ACTIVATE = . $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python3

.PHONY: env
env: $(VENV_NAME)/bin/activate
$(VENV_NAME)/bin/activate: requirements.txt requirements-tf.txt
	test -d $(VENV_NAME) || python3 -m venv $(VENV_NAME)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-tf.txt
	$(PYTHON) -m pip install -e .
	$(VENV_NAME)/bin/pre-commit install
	touch $(VENV_NAME)/bin/activate

##########
# TASKS
##########
.PHONY: train-classification
train-classification: env
	$(PYTHON) -m pipeline.tasks.classification.train --config $(config)

######
# MISC
######
.PHONY: test
test: env
	$(VENV_ACTIVATE) && \
	pytest \
		tests \
		--durations=0
