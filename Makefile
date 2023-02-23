PYTHON_MODULE_PATH=instructmultilingual

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -delete
	find . -name ".ipynb_checkpoints" -type d -delete

format:
	isort --verbose ${PYTHON_MODULE_PATH}
	yapf --verbose --in-place --recursive ${PYTHON_MODULE_PATH} --style='{based_on_style: google, indent_width: 4, column_limit: 120}'
	autoflake --in-place  --remove-all-unused-imports --expand-star-imports --ignore-init-module-imports -r ${PYTHON_MODULE_PATH}
	docformatter --in-place --recursive ${PYTHON_MODULE_PATH}

pylinting:
	## https://vald-phoenix.github.io/pylint-errors/
	pylint --output-format=colorized ${PYTHON_MODULE_PATH}

all:
	@$(MAKE) clean
	@$(MAKE) format
	@$(MAKE) pylinting
