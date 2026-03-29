.PHONY: up down ingest serve test eval deploy-kind clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## Start the full stack (Milvus + Server + Frontend + Ingestion)
	docker-compose up -d milvus-etcd milvus-minio milvus
	@echo "Waiting for Milvus to be healthy..."
	@until docker inspect --format='{{.State.Health.Status}}' milvus-standalone 2>/dev/null | grep -q healthy; do sleep 3; done
	docker-compose up -d server frontend
	@echo "Running ingestion pipeline..."
	docker-compose run --rm ingestion
	@echo ""
	@echo "=== Ready ==="
	@echo "Frontend: http://localhost:8080"
	@echo "API:      http://localhost:8000"
	@echo "MCP:      http://localhost:8001"
	@echo "Health:   curl http://localhost:8000/health"

down: ## Stop everything and remove volumes
	docker-compose down -v

ingest: ## Run the ingestion pipeline only
	docker-compose run --rm ingestion

serve: ## Start server and frontend only (assumes Milvus is running)
	docker-compose up -d server frontend

test: ## Run unit tests
	python -m pytest tests/ -v --tb=short

eval: ## Run the golden dataset evaluation
	python eval/evaluate.py

deploy-kind: ## Deploy to a Kind cluster
	bash scripts/setup-kind.sh

clean: ## Full cleanup: containers, volumes, Kind cluster
	docker-compose down -v
	kind delete cluster --name docs-agent 2>/dev/null || true
	rm -rf ingestion/data/ eval/results/
