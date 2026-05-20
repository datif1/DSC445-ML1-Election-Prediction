# ==========================================================
# DSC445 Machine Learning Project Makefile
# ==========================================================

PYTHON = python

.PHONY: help preprocess classify regress cluster clean

help:
	@echo "Available commands:"
	@echo "  make preprocess  - Run data preprocessing pipeline"
	@echo "  make classify    - Run classification models"
	@echo "  make regress     - Run regression models"
	@echo "  make cluster     - Run clustering analysis"
	@echo "  make clean       - Remove generated report CSV files"

preprocess:
	$(PYTHON) src/preprocessing.py

classify:
	$(PYTHON) src/classification_models.py

regress:
	$(PYTHON) src/regression_models.py

cluster:
	$(PYTHON) src/clustering.py

clean:
	del /Q reports\*.csv 2>nul || exit 0