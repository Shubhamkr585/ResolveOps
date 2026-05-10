# ResolveOps: Autonomous Customer Resolution Agent Engine

ResolveOps is a production-grade autonomous agent engine built with **LangGraph**, **FastAPI**, **PostgreSQL**, and **Redis**. Designed in strict accordance with the **LangGraph Agent Engineering Handbook**, it implements deterministic state transitions, multi-tier memory management (short-term PostgreSQL checkpointing and long-term customer Store), context-budget engineering, distributed tool safety (mutex locking + idempotency), and cryptographic Human-in-the-Loop (HITL) approval workflows.

---

## Key Highlights

- **Zero Hardcoding**: All business limits, return eligibility windows, and product policies are retrieved dynamically from PostgreSQL using Full-Text Search (`ts_rank` + `plainto_tsquery`).
- **Real Database Transactions**: Real atomic mutations update orders to `REFUNDED` and write immutable ledger records to `refund_transactions` with `SELECT ... FOR UPDATE` isolation.
- **Self-Bootstrapping Database**: An automated startup bootstrapper (`core/db_bootstrap.py`) idempotently initializes all tables, full-text indexes, official policies, and a 150-order dataset upon boot on any cloud PostgreSQL instance (Neon, Supabase, Render, AWS RDS).
- **Interactive Interview Dashboard**: A modern web interface served at `/dashboard` with a live Customer Chat simulator, Supervisor HITL Desk, live PostgreSQL data inspector, and Redis mutex monitoring.
- **Two-Tier Distributed Safety**: Redis distributed locking (`lock:entity:{order_id}`) and deterministic UUIDv5 idempotency keys (`idempotency:{action_id}`) prevent double-refunds during network retries.
- **Enterprise Observability**: Structured JSON audit logging with correlation IDs (`thread_id`, `run_id`, `task_id`), execution timers, and Prometheus/Kubernetes liveness (`/healthz`) and readiness (`/readyz`) probes.

---

## System Architecture

```
START
  |
  v
[guardrail_input] --------> (Detects prompt injection and redacts PII)
  |
  v
[triage] -----------------> (Classifies intent and sentiment; dual-model fallback)
  |
  v
[context_loader] ---------> (Loads durable customer memory from Store; prunes context window)
  |
  v
[investigation] ----------> (Executes read-only tools: DB orders, DB customer history, DB policies FTS)
  |
  v
<check_investigation_sufficiency>
  |-- (Missing Order ID) ------> [ask_clarifying]
  |-- (Order Not Found) -------> [order_not_found]
  |-- (Outside Return Window) -> [order_ineligible]
  |-- (Policy FAQ Question) ---> [policy_rag]
  |-- (Complete Evidence) -----> [resolution_planner]
                                       |
                                       v
                             <check_escalation_guard>
                               /                   \
               (Amount > Policy Limit)        (Amount <= Policy Limit)
                             /                       \
             [hitl_suspension]              [tool_execution]
             (Native interrupt)             (Atomic DB Mutation + Redis Lock)
                     |                               |
       (HMAC Signed Admin Webhook)                   |
                     |                               |
                     +-------------------------------+
                     |
                     v
             [verifier_and_memory] -> (Persists dispute summary in Postgres Store)
                     |
                     v
                    END
```

---

## Memory & Context Architecture

ResolveOps separates memory into three distinct scopes:

1. **Working Memory**: In-flight state attributes (`messages`, `investigation_evidence`, `pending_action`, `action_results`) residing in graph state during execution.
2. **Short-Term Memory**: Multi-turn dialogue history checkpointed across user sessions into PostgreSQL via `AsyncPostgresSaver`.
3. **Long-Term Memory**: Cross-thread durable customer facts (account tier, dispute history, settlement counts) persisted across threads in PostgreSQL via `AsyncPostgresStore` under the `("customers", customer_id)` namespace.
4. **Context Budget Management**: A sliding token-budget window retains the most recent 6 verbatim messages, compressing older turns into an evolving rolling summary.

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Docker & Docker Compose (or local PostgreSQL and Redis)

### 2. Environment Setup
Copy the environment template:
```bash
cp .env.example .env
```

Provide your `GEMINI_API_KEY` in `.env` (optional; deterministic heuristics run if omitted).

### 3. Start Database & Redis via Docker
```bash
docker-compose up -d
```
*Note: Starts PostgreSQL on port `5433` and Redis on port `6379`.*

### 4. Install Dependencies
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Run the Server
```bash
python main.py
```
The application will automatically bootstrap tables, full-text indexes, and seed catalog data on startup.

---

## Interactive Interview Dashboard

Open your browser to:
```
http://127.0.0.1:8000/dashboard
```

The presentation dashboard includes:
1. **Live Customer Chat**: Test scenarios including high-value refunds, under-threshold refunds, missing order ID inquiries, and prompt injection attacks.
2. **Supervisor HITL Approval Desk**: Displays active suspended actions awaiting human authorization with 1-click HMAC-signed "Approve" and "Reject" triggers.
3. **Live System of Record**: View PostgreSQL `orders`, `refund_transactions`, and `customers` in real time with auto-refresh.
4. **Health & OpenAPI Links**: Direct access to `/docs` and `/readyz`.

---

## Production Cloud Deployment

### Universal Docker Container
Build and run the production container:
```bash
docker build -t resolveops:latest .
docker run -p 8000:8000 --env-file .env resolveops:latest
```

### Full Production Stack with Docker Compose
```bash
docker compose -f docker-compose.prod.yml up -d
```

### 1-Click Deploy on Render / Railway
This repository contains `render.yaml` for 1-click deployment on Render. The application auto-detects SSL connection strings (`sslmode=require`) from managed PostgreSQL providers like **Neon**, **Supabase**, and **Render**.

---

## API Reference

### 1. Customer Chat
- **POST** `/v1/chat/message`
- **Request Body**:
  ```json
  {
    "message": "I demand an immediate refund for order ORD-9821 of $120. It arrived broken.",
    "customer_id": "CUST-1001",
    "thread_id": "optional-uuid"
  }
  ```
- **Response** (When exceeding policy limit):
  ```json
  {
    "thread_id": "b9214804-c0ee-4192-8528-424c279f7df9",
    "status": "SUSPENDED_FOR_APPROVAL",
    "reply": "Your refund request for order ORD-9821 ($120.00) exceeds our automated approval limit ($100.00)...",
    "interruption_details": {
      "type": "MUTATION_APPROVAL_REQUIRED",
      "action_id": "e893d146-4972-5073-9964-c7a1d6de24e8",
      "action_type": "refund_order",
      "amount": 120.0,
      "entity_id": "ORD-9821"
    }
  }
  ```

### 2. Supervisor Webhook Resume
- **POST** `/v1/admin/webhooks/resume-action`
- **Headers**:
  - `X-ResolveOps-Signature`: `sha256=<HMAC-SHA256 signature of body using ADMIN_WEBHOOK_SECRET>`
- **Request Body**:
  ```json
  {
    "thread_id": "b9214804-c0ee-4192-8528-424c279f7df9",
    "action_id": "e893d146-4972-5073-9964-c7a1d6de24e8",
    "decision": "APPROVED",
    "admin_id": "ADMIN-SUPERVISOR-1",
    "override_reason": "Verified damaged photos"
  }
  ```

### 3. Observability & Health Probes
- **GET** `/healthz` - Liveness probe (returns `200 OK`)
- **GET** `/readyz` - Readiness probe (checks active PostgreSQL and Redis socket connectivity)

---

## Running the Test Suite

```bash
pytest -v
```
All integration tests verify the end-to-end LangGraph cycle, prompt injection guardrails, dynamic clarification, and HMAC webhook resumption.
