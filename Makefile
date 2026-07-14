PYTHON ?= python
.PHONY: test demo evaluate dataset
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
demo:
	PYTHONPATH=src $(PYTHON) -m pixel_to_program demo --output artifacts/demo.json
evaluate:
	PYTHONPATH=src $(PYTHON) -m pixel_to_program evaluate --output artifacts/evaluation.json
dataset:
	PYTHONPATH=src $(PYTHON) -m pixel_to_program dataset --directory data/generated
