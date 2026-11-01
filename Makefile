ENV ?= dev
PROFILE ?= dev
CONFIG := project_config.yml

# Read catalog/schema from project_config.yml for ENV
CATALOG := $(shell uv run python -c "import yaml; print(yaml.safe_load(open('$(CONFIG)'))['$(ENV)']['catalog'])")
SCHEMA  := $(shell uv run python -c "import yaml; print(yaml.safe_load(open('$(CONFIG)'))['$(ENV)']['schema'])")
WAREHOUSE_ID := $(shell uv run python -c "import yaml; print(yaml.safe_load(open('$(CONFIG)'))['$(ENV)']['warehouse_id'])")

# UC volume "packages" + python folder
VOLUME_NAME := packages
VOLUME_FQN  := $(CATALOG).$(SCHEMA).$(VOLUME_NAME)
PACKAGES_DIR := /Volumes/$(CATALOG)/$(SCHEMA)/$(VOLUME_NAME)/python

.PHONY: build build_publish build_publish-dev build_publish-acc build_publish-prd

build:
	uv build

build_publish: build
	@echo "Publishing wheel → $(PACKAGES_DIR) (env=$(ENV))"
	# Ensure UC volume exists
	-databricks volumes create $(CATALOG) $(SCHEMA) $(VOLUME_NAME) MANAGED --profile $(PROFILE)

	# Ensure packages folder exists, then copy latest wheel
	databricks fs mkdirs "dbfs:$(PACKAGES_DIR)" --profile $(PROFILE)
	@whl=$$(ls -t dist/*.whl | head -n1); \
	echo "Copying $$whl"; \
	databricks fs cp "$$whl" "dbfs:$(PACKAGES_DIR)/" --profile $(PROFILE) --overwrite