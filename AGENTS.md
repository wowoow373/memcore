# Project Guidelines

## Code Style
- Python: use ruff and isort (black profile) through Make targets.
- TypeScript: use prettier in mem0-ts.
- Keep changes scoped to the package you are touching; avoid cross-package edits unless required.
- Prefer config-driven initialization patterns already used in the repo (for example, Memory.from_config in Python).

## Architecture
- Monorepo with multiple packages:
  - mem0/: core Python SDK
  - mem0-ts/: TypeScript SDK
  - server/: FastAPI self-hosted REST API
  - docs/: Mintlify docs source
- Core backend design is pluggable and factory-based (LLM, embedder, vector store, graph store, reranker).
- Memory add flow in Python is graph-based (LangGraph state machine). Changes in add flow should consider shared state and node interactions.
- Legacy or experimental areas exist (embedchain, openmemory) and are not part of normal lint scope.

## Build and Test
- Python SDK setup and checks:
  - hatch shell dev_py_3_11
  - make format
  - make sort
  - make lint
  - make test
- Python multi-version checks:
  - make test-py-3.9
  - make test-py-3.10
  - make test-py-3.11
  - make test-py-3.12
- TypeScript SDK (from mem0-ts/):
  - pnpm install
  - pnpm run format
  - pnpm run build
  - pnpm run test:unit
  - pnpm run test:integration
- Server (from server/):
  - make build
  - make run_local
  - or uvicorn main:app --reload for direct local run

## Conventions and Pitfalls
- Prefer relative paths while working in this repo.
- Validate Docker state before API verification:
  - docker compose ps
- For server log-driven debugging, use targeted tails:
  - docker compose logs -f mem0 --tail 20
- For local server auth tests, use X-API-Key from server/.env.
- Do not duplicate long explanations already documented; link to source docs.

## Reference Docs (Link, Do Not Duplicate)
- Contributor workflow: CONTRIBUTING.md
- Development notes and architecture context: CLAUDE.md
- Open source docs index: docs/README.md
- Python quickstart: docs/open-source/python-quickstart.mdx
- Node quickstart: docs/open-source/node-quickstart.mdx
- Configuration: docs/open-source/configuration.mdx
- Feature docs: docs/open-source/features/
- Core concepts: docs/core-concepts/
- Server usage: server/README.md
- TS SDK usage: mem0-ts/README.md

## Scope Guidance
- Put repo-wide defaults here.
- If future work needs different behavior per area, add nested AGENTS.md files in package folders (for example mem0/, mem0-ts/, server/).
