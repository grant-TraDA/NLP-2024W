SHELL = /bin/zsh # please modify this line to your shell path

CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: help
help:
	@echo "Commands:"
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."

.PHONY: style
style:
	($(CONDA_ACTIVATE) mini-rag-dev ; ruff format . && ruff check .)

.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -rf .coverage*
