import os
import re
import uuid
import logging
from typing import Dict, Any, Literal, Optional, List
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore

from core.config import get_settings
from core.state import ResolveOpsState, TriageDecision, PendingAction
from core.decorators import LockAcquisitionError, ActionInProgressError
from core.guardrails import sanitize_and_check_injection, validate_refund_policy
from core.memory import MemoryManager
from core.observability import AuditLogger, ExecutionTimer
from tools.order import get_order_details, get_customer_history
from tools.policy import search_refund_policy
from tools.refund import process_refund_tool

logger = logging.getLogger("resolveops.graph")
settings = get_settings()

# Model Configuration with Resilience Defaults (Handbook Section 8.3)
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=1
) if settings.GEMINI_API_KEY else None

fallback_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_FALLBACK_MODEL,
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=1
) if settings.GEMINI_API_KEY and settings.GEMINI_MODEL != settings.GEMINI_FALLBACK_MODEL else None

TRIAGE_PROMPT = """You are the triage supervisor for an e-commerce resolution system.
Analyze the user dialogue and classify intent into exactly one category:
- 'policy_rag': General questions, return timelines, shipping FAQ.
- 'order_management': Track, cancel, or modify active order addresses.
- 'billing_refund': Disputes, refund requests, charging errors.
- 'human_escalation': Highly abusive speech or legal threats.

Extract sentiment (-1.0 to 1.0). If sentiment < -0.6, set requires_escalation=True."""


def extract_order_id_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """Scans conversation history in reverse to extract mentioned order ID."""
    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        match = re.search(r"\b(ORD-\d+|[A-Z]{3,4}-\d+)\b", content, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def extract_explicit_refund_amount(text: str) -> Optional[float]:
    """Extracts explicit numeric refund quantity if requested by customer."""
    m = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d+(?:\.\d{1,2})?)\s*(?:dollars|usd|\$)\b", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"refund\s+(?:of\s+)?(\d+(?:\.\d{1,2})?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def heuristic_triage(state: ResolveOpsState) -> TriageDecision:
    """Deterministic fallback triage when LLM is unavailable or times out."""
    last_msg = state["messages"][-1].content.lower() if state.get("messages") else ""
    if any(k in last_msg for k in ["refund", "money", "broken", "damaged", "return", "chargeback"]):
        return TriageDecision(
            route="billing_refund",
            sentiment_score=-0.5,
            requires_escalation=False,
            reason="Refund or damaged item intent detected (deterministic fallback)"
        )
    return TriageDecision(
        route="policy_rag",
        sentiment_score=0.1,
        requires_escalation=False,
        reason="General inquiry (deterministic fallback)"
    )


# ---------------------------------------------------------------------------
# Node 1: Input Guardrails & Sanitization (Handbook Section 8.4, 8.5)
# ---------------------------------------------------------------------------
async def guardrail_input_node(state: ResolveOpsState) -> Dict[str, Any]:
    with ExecutionTimer() as timer:
        raw_msg = state["messages"][-1].content if state.get("messages") else ""
        check = sanitize_and_check_injection(raw_msg)

    if not check.is_safe:
        AuditLogger.log_event(
            event_type="GUARDRAIL_VIOLATION",
            thread_id=state["session_id"],
            customer_id=state["customer_id"],
            status="SECURITY_VIOLATION",
            latency_ms=timer.elapsed_ms,
            payload={"violation": check.violation_type, "metadata": check.metadata}
        )
        return {
            "is_escalated": True,
            "escalation_reason": "Prompt injection attempt detected.",
            "messages": [AIMessage(content="Security Policy: Your input contained unauthorized instructions and was blocked.")]
        }

    AuditLogger.log_event(
        event_type="GUARDRAIL_CHECK_PASSED",
        thread_id=state["session_id"],
        customer_id=state["customer_id"],
        latency_ms=timer.elapsed_ms
    )
    return {}


# ---------------------------------------------------------------------------
# Node 2: Triage Classification (Handbook Section 1.4, 7.1)
# ---------------------------------------------------------------------------
async def triage_node(state: ResolveOpsState) -> Dict[str, Any]:
    if state.get("is_escalated") and state.get("escalation_reason") == "Prompt injection attempt detected.":
        return {"current_route": "human_escalation"}

    decision: Optional[TriageDecision] = None
    with ExecutionTimer() as timer:
        if llm:
            try:
                structured_llm = llm.with_structured_output(TriageDecision)
                decision = await structured_llm.ainvoke(
                    [SystemMessage(content=TRIAGE_PROMPT)] + list(state["messages"])
                )
            except Exception as exc:
                logger.warning(f"Primary model '{settings.GEMINI_MODEL}' failed: {exc}. Attempting fallback...")
                if fallback_llm:
                    try:
                        fallback_structured = fallback_llm.with_structured_output(TriageDecision)
                        decision = await fallback_structured.ainvoke(
                            [SystemMessage(content=TRIAGE_PROMPT)] + list(state["messages"])
                        )
                    except Exception as fb_exc:
                        logger.warning(f"Fallback model failed: {fb_exc}. Using deterministic heuristic.")

    if not decision:
        decision = heuristic_triage(state)

    AuditLogger.log_event(
        event_type="TRIAGE_DECISION",
        thread_id=state["session_id"],
        customer_id=state["customer_id"],
        latency_ms=timer.elapsed_ms,
        payload={"route": decision.route, "sentiment": decision.sentiment_score, "reason": decision.reason}
    )

    return {
        "triage": decision,
        "current_route": decision.route,
        "is_escalated": decision.requires_escalation,
        "escalation_reason": "Low sentiment detected" if decision.requires_escalation else None
    }


# ---------------------------------------------------------------------------
# Node 3: Context Loader & Long-Term Memory (Handbook Section 4.3, 5.3)
# ---------------------------------------------------------------------------
async def context_loader_node(state: ResolveOpsState, *, store: Optional[BaseStore] = None) -> Dict[str, Any]:
    with ExecutionTimer() as timer:
        customer_memory = await MemoryManager.retrieve_long_term_memory(store, state["customer_id"])
        pruned_messages, rolling_summary = MemoryManager.apply_context_window_budget(
            list(state["messages"]),
            state.get("summary")
        )

    AuditLogger.log_event(
        event_type="CONTEXT_LOADED",
        thread_id=state["session_id"],
        customer_id=state["customer_id"],
        latency_ms=timer.elapsed_ms,
        payload={"tier": customer_memory.get("profile", {}).get("tier", "Standard"), "has_summary": bool(rolling_summary)}
    )

    return {
        "customer_memory": customer_memory,
        "summary": rolling_summary
    }


# ---------------------------------------------------------------------------
# Node 4: Investigation with Read-Only Database Tools (Handbook Section 6.4, 11.2)
# ---------------------------------------------------------------------------
async def investigation_node(state: ResolveOpsState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content if state.get("messages") else ""
    order_id = extract_order_id_from_messages(list(state.get("messages", [])))

    with ExecutionTimer() as timer:
        order_evidence: Dict[str, Any] = {}
        if order_id:
            order_evidence = await get_order_details(order_id=order_id, customer_id=state["customer_id"])

        policy_evidence = await search_refund_policy(last_msg)
        customer_evidence = await get_customer_history(customer_id=state["customer_id"])

    evidence = [
        {"source": "get_customer_history", "data": customer_evidence},
        {"source": "search_refund_policy", "data": policy_evidence}
    ]
    if order_evidence:
        evidence.append({"source": "get_order_details", "data": order_evidence})

    AuditLogger.log_event(
        event_type="READ_TOOLS_EXECUTED",
        thread_id=state["session_id"],
        customer_id=state["customer_id"],
        latency_ms=timer.elapsed_ms,
        payload={
            "order_id_detected": order_id,
            "order_found": order_evidence.get("found", False) if order_evidence else False,
            "customer_orders_count": customer_evidence.get("total_orders", 0)
        }
    )

    return {"investigation_evidence": evidence}


# ---------------------------------------------------------------------------
# Conditional Branch after Investigation
# ---------------------------------------------------------------------------
def check_investigation_sufficiency(state: ResolveOpsState) -> Literal["ask_clarifying", "policy_rag", "order_ineligible", "order_not_found", "propose_resolution", "security_abort"]:
    if state.get("escalation_reason") == "Prompt injection attempt detected.":
        return "security_abort"

    if state.get("current_route") == "policy_rag":
        return "policy_rag"

    order_id = extract_order_id_from_messages(list(state.get("messages", [])))
    if not order_id:
        return "ask_clarifying"

    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    order_data = evidence_map.get("get_order_details", {})

    if not order_data.get("found"):
        return "order_not_found"

    if not order_data.get("is_return_eligible"):
        return "order_ineligible"

    return "propose_resolution"


# ---------------------------------------------------------------------------
# Node 5a: Clarification Node
# ---------------------------------------------------------------------------
async def ask_clarifying_node(state: ResolveOpsState) -> Dict[str, Any]:
    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    cust_data = evidence_map.get("get_customer_history", {})
    recent_orders = cust_data.get("recent_orders", [])

    if recent_orders:
        orders_summary = "\n".join([f"- {o['order_id']} (${o['total']:.2f}, {o['status']})" for o in recent_orders[:3]])
        reply = (
            f"I can help you with your refund or return request. To make sure I assist with the correct purchase, "
            f"could you please specify which order you would like to resolve?\n\n"
            f"Here are your recent orders:\n{orders_summary}\n\n"
            f"Please reply with the Order ID (e.g. {recent_orders[0]['order_id']}) and the reason."
        )
    else:
        reply = (
            "I can assist you with your refund inquiry. Could you please provide your Order ID "
            "and a brief description of the issue?"
        )

    return {"messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Node 5b: Order Not Found / Policy Ineligible Nodes
# ---------------------------------------------------------------------------
async def order_not_found_node(state: ResolveOpsState) -> Dict[str, Any]:
    order_id = extract_order_id_from_messages(list(state.get("messages", []))) or "your order"
    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    order_data = evidence_map.get("get_order_details", {})
    message = order_data.get("message") or f"We could not locate order '{order_id}' under your account. Please verify the order number."
    return {"messages": [AIMessage(content=message)]}


async def order_ineligible_node(state: ResolveOpsState) -> Dict[str, Any]:
    order_id = extract_order_id_from_messages(list(state.get("messages", []))) or "your order"
    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    policy_data = evidence_map.get("search_refund_policy", {})
    window = policy_data.get("return_window_days", settings.RETURN_WINDOW_DAYS_DEFAULT)

    reply = (
        f"Order '{order_id}' is outside our {window}-day return eligibility window. "
        f"Under company policy ({policy_data.get('title', 'Resolution Policy')}), automated refunds cannot be processed for expired orders."
    )
    return {"messages": [AIMessage(content=reply)]}


# ---------------------------------------------------------------------------
# Node 5c: Resolution Planner
# ---------------------------------------------------------------------------
async def resolution_planner_node(state: ResolveOpsState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content if state.get("messages") else ""
    order_id = extract_order_id_from_messages(list(state.get("messages", [])))

    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    order_data = evidence_map.get("get_order_details", {})
    order_total = float(order_data.get("order_total", 0.0))

    explicit_amount = extract_explicit_refund_amount(last_msg)
    if explicit_amount is not None:
        refund_amount = min(explicit_amount, order_total)
    else:
        refund_amount = order_total

    reason = "Customer requested refund"
    if "broken" in last_msg.lower() or "damaged" in last_msg.lower():
        reason = "Item arrived damaged/broken"
    elif "late" in last_msg.lower() or "delay" in last_msg.lower():
        reason = "Item arrived late"

    idempotency_key = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{state['session_id']}-{order_id}-refund"))
    action = PendingAction(
        action_id=idempotency_key,
        action_type="refund_order",
        entity_id=order_id or "UNKNOWN",
        amount=refund_amount,
        reason=reason,
        metadata={"customer_id": state["customer_id"], "order_total": order_total}
    )

    return {"pending_action": action}


# ---------------------------------------------------------------------------
# Routing Guard: Escalation & Policy Guard (Dynamic Threshold from DB Policy)
# ---------------------------------------------------------------------------
def check_escalation_guard(state: ResolveOpsState) -> Literal["suspend_for_hitl", "execute_mutation"]:
    action = state.get("pending_action")
    if not action:
        return "execute_mutation"

    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    policy_data = evidence_map.get("search_refund_policy", {})
    policy_limit = float(policy_data.get("max_automated_limit", settings.MAX_AUTO_REFUND_LIMIT_DEFAULT))

    # Business Rule: Transactions exceeding policy limit require human supervisor sign-off
    if action.action_type == "refund_order" and action.amount > policy_limit:
        return "suspend_for_hitl"

    if state.get("is_escalated", False):
        return "suspend_for_hitl"

    return "execute_mutation"


# ---------------------------------------------------------------------------
# Node 6: Human-in-the-Loop Suspension
# ---------------------------------------------------------------------------
def hitl_suspension_node(state: ResolveOpsState) -> Dict[str, Any]:
    action = state["pending_action"]
    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    policy_data = evidence_map.get("search_refund_policy", {})
    policy_limit = float(policy_data.get("max_automated_limit", settings.MAX_AUTO_REFUND_LIMIT_DEFAULT))

    human_approval = interrupt({
        "type": "MUTATION_APPROVAL_REQUIRED",
        "action_id": action.action_id,
        "action_type": action.action_type,
        "amount": action.amount,
        "entity_id": action.entity_id,
        "reason": action.reason,
        "customer_id": state["customer_id"],
        "policy_limit": policy_limit,
        "message": (
            f"Your refund request for order {action.entity_id} (${action.amount:.2f}) exceeds our automated "
            f"approval limit (${policy_limit:.2f}) and has been submitted to a human supervisor for review."
        )
    })
    return {
        "admin_approval_status": human_approval.get("status", "REJECTED"),
        "is_escalated": False
    }


# ---------------------------------------------------------------------------
# Node 7: Action Executor (PostgreSQL Mutation + Redis Locks)
# ---------------------------------------------------------------------------
async def tool_execution_node(state: ResolveOpsState) -> Dict[str, Any]:
    if state.get("admin_approval_status") == "REJECTED":
        return {
            "action_results": [{"status": "ABORTED", "message": "Transaction rejected by supervisor."}],
            "messages": [AIMessage(content="Your refund exceeds auto-approval limits and was declined by supervisor review.")]
        }

    action = state["pending_action"]
    with ExecutionTimer() as timer:
        try:
            res = await process_refund_tool(
                order_id=action.entity_id,
                action_id=action.action_id,
                amount=action.amount,
                reason=action.reason,
                customer_id=state["customer_id"]
            )
            AuditLogger.log_event(
                event_type="TRANSACTIONAL_TOOL_EXECUTED",
                thread_id=state["session_id"],
                customer_id=state["customer_id"],
                latency_ms=timer.elapsed_ms,
                payload={"action_id": action.action_id, "amount": action.amount, "status": res.get("status")}
            )
            return {
                "action_results": [res],
                "messages": [AIMessage(content=f"Successfully refunded ${action.amount:.2f} for order {action.entity_id}. Transaction ID: {res.get('transaction_id')}.")]
            }
        except LockAcquisitionError:
            return {"error_trace": "Resource contention lock held."}
        except ActionInProgressError:
            return {"error_trace": "Duplicate in-flight execution."}
        except Exception as exc:
            logger.error(f"Tool execution failed: {exc}")
            return {
                "action_results": [{"status": "FAILED", "error": str(exc)}],
                "messages": [AIMessage(content=f"An error occurred while processing refund: {str(exc)}")]
            }


# ---------------------------------------------------------------------------
# Node 8: Policy RAG Response (Full-Text Search Grounded)
# ---------------------------------------------------------------------------
async def policy_rag_node(state: ResolveOpsState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content if state.get("messages") else ""
    policy_info = await search_refund_policy(last_msg)
    title = policy_info.get("title", "Policy Information")
    policy_text = policy_info.get("content", "Standard resolution policy applies.")
    response_text = f"**{title}**\n\n{policy_text}"
    return {"messages": [AIMessage(content=response_text)]}


# ---------------------------------------------------------------------------
# Node 9: Verifier & Long-Term Memory Updater
# ---------------------------------------------------------------------------
async def verifier_and_memory_node(state: ResolveOpsState, *, store: Optional[BaseStore] = None) -> Dict[str, Any]:
    action_results = state.get("action_results", [])
    if action_results and action_results[-1].get("status") == "SETTLED":
        action = state.get("pending_action")
        if action and store:
            await MemoryManager.persist_long_term_fact(
                store=store,
                customer_id=state["customer_id"],
                fact_key="dispute_summary",
                fact_value={
                    "total_refunds_count": 1,
                    "last_refund_order": action.entity_id,
                    "last_refund_amount": action.amount
                }
            )

    return {}


# ---------------------------------------------------------------------------
# StateGraph Assembly
# ---------------------------------------------------------------------------
def build_resolveops_graph(checkpointer: AsyncPostgresSaver, store: Optional[BaseStore] = None):
    workflow = StateGraph(ResolveOpsState)

    workflow.add_node("guardrail_input", guardrail_input_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("context_loader", context_loader_node)
    workflow.add_node("investigation", investigation_node)
    workflow.add_node("ask_clarifying", ask_clarifying_node)
    workflow.add_node("order_not_found", order_not_found_node)
    workflow.add_node("order_ineligible", order_ineligible_node)
    workflow.add_node("resolution_planner", resolution_planner_node)
    workflow.add_node("hitl_suspension", hitl_suspension_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("policy_rag", policy_rag_node)
    workflow.add_node("verifier_and_memory", verifier_and_memory_node)

    # Deterministic control flow
    workflow.add_edge(START, "guardrail_input")
    workflow.add_edge("guardrail_input", "triage")
    workflow.add_edge("triage", "context_loader")
    workflow.add_edge("context_loader", "investigation")

    workflow.add_conditional_edges(
        "investigation",
        check_investigation_sufficiency,
        {
            "ask_clarifying": "ask_clarifying",
            "order_not_found": "order_not_found",
            "order_ineligible": "order_ineligible",
            "policy_rag": "policy_rag",
            "propose_resolution": "resolution_planner",
            "security_abort": END
        }
    )

    workflow.add_conditional_edges(
        "resolution_planner",
        check_escalation_guard,
        {
            "suspend_for_hitl": "hitl_suspension",
            "execute_mutation": "tool_execution"
        }
    )

    workflow.add_edge("ask_clarifying", "verifier_and_memory")
    workflow.add_edge("order_not_found", "verifier_and_memory")
    workflow.add_edge("order_ineligible", "verifier_and_memory")
    workflow.add_edge("hitl_suspension", "tool_execution")
    workflow.add_edge("tool_execution", "verifier_and_memory")
    workflow.add_edge("policy_rag", "verifier_and_memory")
    workflow.add_edge("verifier_and_memory", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
