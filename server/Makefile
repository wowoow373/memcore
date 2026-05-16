infra_up:
	docker compose up -d postgres neo4j

infra_down:
	docker compose down

infra_logs:
	docker compose logs -f --tail 50 postgres neo4j

run_local:
	uvicorn main:app --host 0.0.0.0 --port 8888 --reload

.PHONY: infra_up infra_down infra_logs run_local
