# ResolveOps: Autonomous Customer Support Resolution Engine

ResolveOps is a production-grade autonomous agent engine designed in alignment with the **LangGraph + Agent Engineering Handbook**. It provides deterministic guardrails, multi-tier memory management (short-term thread checkpointing and long-term cross-session storage), context-budget engineering, distributed tool safety, and structured observability.

---

## 1. Architecture Overview

ResolveOps implements the Capstone Architecture (*ResolveAI*) specified in Section 11.2 and Page 16 of the Agent Engineering Handbook.

```
START
  |
  v
[guardrail_input] --------> (Detects prompt injection and redacts PII)
  |
  v
[triage] -----------------> (Classifies intent and sentiment; resilient fallback)
  |
  v
[context_loader] ---------> (Loads long-term customer memory from Store; prunes context window)
  |
  v
[investigation] ----------> (Executes read-only tools: get_order_details, search_refund_policy)
  |
  v
[resolution_planner] -----> (Synthesizes evidence and constructs deterministic PendingAction)
  |
  v
<check_escalation_guard> -> (Applies deterministic threshold checks: amount > $100.00?)
  |                      |
  | (> $100.00)          | (<= $100.00)
  v                      v
[hitl_suspension]      [tool_execution] (Protected by Redis Distributed Mutex & Idempotency)
  |                      |
  +----------------------+
  |
  v
[verifier_and_memory] ----> (Verifies settlement, updates rolling summary, persists durable facts)
  |
  v
 END
```

---

## 2. Memory Architecture

ResolveOps separates memory into three distinct scopes as mandated by Section 4 of the Handbook:

### 2.1 Working Memory (Graph State)
- **Scope**: Current execution run.
- **Implementation**: `ResolveOpsState` (`core/state.py`).
- **Data Held**: Transient task variables, intermediate tool findings (`investigation_evidence`), routing decisions, and execution timers.

### 2.2 Short-Term Memory (Thread Checkpointer)
- **Scope**: Conversation turns within a specific thread (`thread_id`).
- **Implementation**: `AsyncPostgresSaver` backed by PostgreSQL connection pool (`main.py`).
- **Capabilities**: Enables human-in-the-loop suspension (`interrupt()`), durable state snapshots, and fault tolerance across server restarts.

### 2.3 Long-Term Memory (Cross-Thread Store)
- **Scope**: Durable customer profile, preferences, and dispute statistics across independent threads.
- **Implementation**: `AsyncPostgresStore` (`core/memory.py`).
- **Namespace Strategy**: Scoped under `("customers", customer_id)`.
- **Write Policy**: Only durable, verified outcomes (such as settled refunds, loyalty tiers, and communication preferences) are persisted. Unverified conversational noise is excluded.

---

## 3. Context Engineering & Token-Budget Management

To prevent context-window overflow and degradation (Handbook Section 5):
- **Sliding Window**: The most recent 6 messages (`MAX_VERBATIM_MESSAGES`) are retained verbatim.
- **Rolling Compact Summary**: Turns exceeding the verbatim limit are compressed into a structured summary (`goal`, `decisions`, `evidence_refs`) and passed to downstream nodes.
- **Tool Output Hygiene**: Read-only tools return bounded summaries rather than raw database dumps.

---

## 4. Deterministic Guardrails & Security

Prompts are guidance; deterministic code provides security boundaries (Handbook Section 8.4):

1. **Prompt Injection Defense (`core/guardrails.py`)**:
   - Detects adversarial patterns such as instruction overrides, database dump requests, and role impersonation before reaching model nodes.
   - Malicious inputs are immediately diverted to a security abort path.

2. **PII Masking**:
   - Redacts credit card numbers and Social Security numbers prior to state persistence.

3. **Authorization & Tenant Isolation**:
   - Read-only tools verify customer authorization at the service layer (`customer_id` check in `tools/order.py`).

4. **Human-in-the-Loop (HITL) Threshold**:
   - Refund requests exceeding **$100.00** or sessions flagged with negative customer sentiment (< -0.6) automatically pause via `langgraph.types.interrupt()`.
   - Requests below or equal to $100.00 execute automatically.

5. **Distributed Mutex & Idempotency (`core/decorators.py`)**:
   - Write actions (`tools/refund.py`) are decorated with `@transactional_tool`.
   - Uses Redis distributed locking with time-to-live (TTL) and atomic Lua release scripts.
   - Enforces deterministic UUIDv5 idempotency keys to prevent duplicate execution during network retries.

---

## 5. Observability & Audit Logging

Implements Section 9 of the Handbook:
- **Correlation Identifiers**: Every event binds `thread_id`, `customer_id`, `task_id`, and `run_id`.
- **Execution Timers**: Measures millisecond latency for each node transition and tool invocation.
- **Structured JSON Logs**: Emits machine-parsable audit trails for compliance and debugging.

---

## 6. API Reference

### 6.1 Customer Chat
- **Endpoint**: `POST /v1/chat/message`
- **Payload**:
  ```json
  {
    "message": "I want a full refund of $12 for order ORD-9821. It arrived completely broken.",
    "customer_id": "CUST-1001",
    "thread_id": "optional-uuid"
  }
  ```
- **Response (Auto-Approved, <= $100)**:
  ```json
  {
    "thread_id": "0fc6cac9-d07a-4231-b2c7-9cbaf85f06d5",
    "status": "COMPLETED",
    "reply": "Successfully refunded $12.00 for order ORD-9821.",
    "interruption_details": null
  }
  ```
- **Response (Suspended for Review, > $100)**:
  ```json
  {
    "thread_id": "71ad978c-d5ff-4488-a947-f78d763ff486",
    "status": "SUSPENDED_FOR_APPROVAL",
    "reply": "Your refund request exceeds automated limits ($100). It has been sent to a human supervisor for review.",
    "interruption_details": {
      "type": "MUTATION_APPROVAL_REQUIRED",
      "action_id": "b3c6b5d9-3aa6-5094-9d7b-a5a817c9f50c",
      "action_type": "refund_order",
      "amount": 120.0,
      "entity_id": "ORD-9821",
      "reason": "Item arrived damaged/broken"
    }
  }
  ```

### 6.2 Supervisor Approval Webhook
- **Endpoint**: `POST /v1/admin/webhooks/resume-action`
- **Security**: Requires `X-ResolveOps-Signature` header computed as HMAC-SHA256 over raw JSON payload using `ADMIN_WEBHOOK_SECRET`.
- **Payload**:
  ```json
  {
    "thread_id": "71ad978c-d5ff-4488-a947-f78d763ff486",
    "action_id": "b3c6b5d9-3aa6-5094-9d7b-a5a817c9f50c",
    "decision": "APPROVED",
    "admin_id": "SUPERVISOR-42",
    "override_reason": "Verified attached damage photos."
  }
  ```
- **Response**:
  ```json
  {
    "status": "SUCCESS",
    "thread_id": "71ad978c-d5ff-4488-a947-f78d763ff486",
    "action_id": "b3c6b5d9-3aa6-5094-9d7b-a5a817c9f50c",
    "execution_result": {
      "transaction_id": "tx_rf_ORD-9821",
      "order_id": "ORD-9821",
      "refunded_amount": 120.0,
      "status": "SETTLED"
    },
    "detail": "Resumed and executed with status APPROVED."
  }
  ```

### 6.3 Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy", "service": "ResolveOps Engine"}`

---

## 7. Verification & Automated Test Suite

The test suite covers end-to-end paths, guardrails, and memory persistence:

```powershell
.\.venv\Scripts\pytest.exe -v
```

### Verified Test Cases:
1. `test_end_to_end_hitl_refund_flow`: Verifies HITL pause for amounts exceeding $100.00 and resumption via signed supervisor HMAC webhook.
2. `test_auto_approval_under_threshold`: Verifies instant execution for amounts <= $100.00.
3. `test_prompt_injection_guardrail_blocking`: Verifies deterministic blocking of prompt injection attacks.
4. `test_long_term_memory_persistence`: Verifies cross-thread retrieval and post-settlement persistence in `AsyncPostgresStore`.
