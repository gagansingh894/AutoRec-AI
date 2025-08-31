.PHONY: help setup setup-dvc track-dvc push-dvc pull-dvc

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup        - Start Minio using Docker Compose"
	@echo "  make setup-dvc    - Configure DVC remote"
	@echo "  make track-dvc    - Track new data/raw in DVC"
	@echo "  make push-dvc     - Push data to MinIO"
	@echo "  make pull-dvc     - Pull data from MinIO"

# Start MinIO
setup:
	docker-compose -f build/docker-compose.yml up --remove-orphans -d

teardown:
	docker-compose -f build/docker-compose.yml down

# Configure DVC remote
setup-dvc:
	dvc remote add -f -d minio_remote s3://autorec-ai
	dvc remote modify minio_remote endpointurl http://0.0.0.0:9000
	dvc remote modify minio_remote use_ssl false

# Track dataset
track-dvc:
	dvc add data/raw
	git add data/raw.dvc .gitignore
	dvc add data/raw_grouped_by_car_make_model_year
	git add data/raw_grouped_by_car_make_model_year.dvc .gitignore
	dvc add data/car_metadata
	git add data/car_metadata.dvc .gitignore
	git commit -m "feat: track data with DVC" || true

# Push dataset to remote
push-dvc:
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin dvc push

# Pull dataset from remote
pull-dvc:
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin dvc pull