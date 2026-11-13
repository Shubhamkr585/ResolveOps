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
from core.db import execute_order_cancellation

logger = logging.getLogger("resolveops.graph")
settings = get_settings()

# Model Configuration with Resilience Defaults (Handbook Section 8.3)
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0.1,
    google_api_key=settings.GEMINI_API_KEY,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=1
) if settings.GEMINI_API_KEY else None

fallback_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_FALLBACK_MODEL,
    temperature=0.1,
    google_api_key=settings.GEMINI_API_KEY,
    timeout=settings.LLM_TIMEOUT_SECONDS,
    max_retries=1
) if settings.GEMINI_API_KEY and settings.GEMINI_MODEL != settings.GEMINI_FALLBACK_MODEL else None

TRIAGE_PROMPT = """You are the triage supervisor for ResolveOps, an autonomous e-commerce customer support engine.
Analyze the user dialogue and classify intent into exactly one category:
- 'billing_refund': Customer wants a refund, money back, compensation, or dispute for an order or item.
- 'order_management': Customer wants to view/list their orders ("list down all my orders", "my purchase history"), track packages, check delivery status, or modify/cancel orders.
- 'policy_rag': Customer is asking about store policies, warranties, returns rules, shipping FAQs, or terms.
- 'human_escalation': Highly abusive speech or explicit legal threats.

Extract sentiment (-1.0 to 1.0). If sentiment < -0.6, set requires_escalation=True."""

AUTONOMOUS_SYSTEM_PROMPT = """You are ResolveOps Assistant, a professional, empathetic, and highly capable autonomous e-commerce resolution agent.
You have real-time access to the customer's account records, verified order history, and official store policies retrieved from PostgreSQL.

Core Directives:
1. Ground every answer in the provided Evidence below (Customer History, Orders, Policies).
2. If the customer asks to list their orders or asks what they purchased, provide a clean, organized list of their orders with Order ID, Total, Status, Date, Tracking Number, and Items.
3. If the customer asks to track an order or check delivery status, report the exact status, delivery date, and tracking number.
4. If the customer asks to cancel an order:
   - If the order is in PROCESSING status: state clearly that the order has been successfully canceled and the charge voided/refunded.
   - If the order has shipped or is delivered: explain politely that orders already in transit or delivered cannot be canceled, and that they may initiate a return within 30 days after arrival.
   - If the order is not found under their account: inform them that the order number was not located in their account and list their active orders.
5. If the customer wants a refund but did not specify an Order ID, list their recent eligible orders and ask which specific order they want to resolve.
6. If the customer's order is outside the return window, explain the policy gently and mention the return window limit.
7. If the customer asks about store policies (returns, warranty, shipping), answer concisely and directly without dumping large blocks of raw policy text.
8. Be conversational, polite, professional, and helpful. Strictly DO NOT use emojis."""


def extract_text_from_message(msg: Any) -> str:
    """Safely extracts clean string text from BaseMessage or content block."""
    if not msg:
        return ""
    content = msg.content if hasattr(msg, "content") else msg
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts).strip()
    return str(content).strip()


def extract_order_id_from_text(text: str) -> Optional[str]:
    """Extracts explicit order ID mentioned in text."""
    match = re.search(r"\b(ORD-\d+|[A-Z]{3,4}-\d+)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def extract_order_id_from_context(messages: List[BaseMessage]) -> Optional[str]:
    """
    Extracts order ID mentioned in context.
    Strictly checks user messages, never assistant messages.
    If the latest user message asks for a general listing, policy FAQ, or greeting,
    does NOT look back into past turns to avoid sticky context pollution.
    """
    if not messages:
        return None

    # Inspect human messages only
    user_messages = [
        m for m in messages
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"
    ]
    if not user_messages:
        return None

    latest_content = user_messages[-1].content if hasattr(user_messages[-1], "content") else str(user_messages[-1])
    order_id = extract_order_id_from_text(latest_content)
    if order_id:
        return order_id

    # If user message is general query, listing, or FAQ, do NOT look back to old order IDs
    latest_lower = latest_content.lower()
    if any(k in latest_lower for k in [
        "list", "my order", "all order", "show order", "history", "purchases",
        "bought", "policy", "warranty", "shipping", "faq", "hello", "hi", "help"
    ]):
        return None

    # Only look back to recent human turn if user is answering a clarification prompt (e.g. "cancel it", "the broken one", "refund please")
    for msg in reversed(user_messages[:-1]):
        content = msg.content if hasattr(msg, "content") else str(msg)
        found = extract_order_id_from_text(content)
        if found:
            return found

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

    # 1. Check for Order Listing / Tracking / Viewing
    if any(k in last_msg for k in ["list", "my orders", "all orders", "show orders", "past orders", "order history", "purchases", "bought"]):
        return TriageDecision(
            route="order_management",
            sentiment_score=0.1,
            requires_escalation=False,
            reason="Customer requested listing or viewing past orders"
        )
    if any(k in last_msg for k in ["track", "where is", "delivery status", "package", "tracking", "status"]):
        return TriageDecision(
            route="order_management",
            sentiment_score=0.0,
            requires_escalation=False,
            reason="Order tracking or status inquiry"
        )

    # 2. Check for Billing / Refund disputes
    if any(k in last_msg for k in ["refund", "money back", "broken", "damaged", "defective", "chargeback", "cancel and refund"]):
        return TriageDecision(
            route="billing_refund",
            sentiment_score=-0.4,
            requires_escalation=False,
            reason="Refund or damaged item dispute detected"
        )

    # 3. Policy questions
    if any(k in last_msg for k in ["policy", "return window", "warranty", "rule", "terms", "faq", "how do i return", "restocking"]):
        return TriageDecision(
            route="policy_rag",
            sentiment_score=0.1,
            requires_escalation=False,
            reason="Policy inquiry detected"
        )

    # 4. Default
    return TriageDecision(
        route="order_management" if "order" in last_msg else "policy_rag",
        sentiment_score=0.1,
        requires_escalation=False,
        reason="General dialogue"
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
                    [SystemMessage(content=TRIAGE_PROMPT)] + list(state["messages"][-3:])
                )
            except Exception as exc:
                logger.warning(f"Primary model triage failed: {exc}. Attempting fallback...")
                if fallback_llm:
                    try:
                        fallback_structured = fallback_llm.with_structured_output(TriageDecision)
                        decision = await fallback_structured.ainvoke(
                            [SystemMessage(content=TRIAGE_PROMPT)] + list(state["messages"][-3:])
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
    is_general_listing = any(k in last_msg.lower() for k in ["list", "my orders", "all orders", "show orders", "history", "purchases"])

    # If the user is asking for general order listing, do NOT latch onto an old order ID from earlier in the thread
    order_id = None if is_general_listing else extract_order_id_from_context(list(state.get("messages", [])))

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
# Dynamic Routing Condition: Action Mutation vs. Conversational Reasoning
# ---------------------------------------------------------------------------
def check_investigation_sufficiency(state: ResolveOpsState) -> Literal["propose_resolution", "conversational_response", "security_abort"]:
    if state.get("escalation_reason") == "Prompt injection attempt detected.":
        return "security_abort"

    route = state.get("current_route")

    # Only route to mutation planner if user is explicitly pursuing a refund
    if route == "billing_refund":
        last_msg = state["messages"][-1].content if state.get("messages") else ""
        is_general_listing = any(k in last_msg.lower() for k in ["list", "my orders", "all orders"])
        order_id = None if is_general_listing else extract_order_id_from_context(list(state.get("messages", [])))

        if order_id:
            evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
            order_data = evidence_map.get("get_order_details", {})
            # Only plan mutation if order was found and is return eligible
            if order_data.get("found") and order_data.get("is_return_eligible"):
                return "propose_resolution"

    # All inquiries, order listings, tracking, policy FAQs, and clarifications are handled conversationally
    return "conversational_response"


# ---------------------------------------------------------------------------
# Node 5: Autonomous Conversational Reasoner (Context-Aware LLM Response)
# ---------------------------------------------------------------------------
async def conversational_response_node(state: ResolveOpsState) -> Dict[str, Any]:
    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    cust_data = evidence_map.get("get_customer_history", {})
    policy_data = evidence_map.get("search_refund_policy", {})
    order_data = evidence_map.get("get_order_details", {})
    last_msg = state["messages"][-1].content if state.get("messages") else ""

    # Construct context evidence block
    orders_list = cust_data.get("recent_orders", [])
    orders_summary = "\n".join([
        f"- Order {o['order_id']}: ${o['total']:.2f} | Status: {o['status']} | Date: {o.get('date', 'N/A')} | Items: {', '.join([str(it) for it in o['items']]) if o.get('items') else 'Standard Merchandise'} | Tracking: {o.get('tracking_number', 'N/A')} | Return Eligible: {o.get('is_return_eligible')}"
        for o in orders_list
    ]) if orders_list else "No past orders found under this account."

    order_lookup_info = "None requested."
    if order_data:
        if order_data.get("found"):
            order_lookup_info = (
                f"Order ID: {order_data.get('order_id')}\n"
                f"Status: {order_data.get('status')}\n"
                f"Total: ${order_data.get('order_total', 0.0):.2f}\n"
                f"Items: {order_data.get('items', [])}\n"
                f"Tracking: {order_data.get('tracking_number', 'N/A')}\n"
                f"Return Eligible: {order_data.get('is_return_eligible')}\n"
                f"Delivered Date: {order_data.get('delivered_date', 'N/A')}"
            )
        else:
            order_lookup_info = f"Order '{order_data.get('order_id')}' was not found under customer account {state['customer_id']}."

    evidence_context = f"""
[Customer Account Profile]
Customer ID: {state['customer_id']}
Name: {cust_data.get('name', 'Customer')}
Tier: {cust_data.get('tier', 'Standard')}
Trust Score: {cust_data.get('trust_score', 0.95)}
Total Orders: {cust_data.get('total_orders', 0)}

[Customer Registered Orders in Database]
{orders_summary}

[Active Business Policy from Database]
Title: {policy_data.get('title', 'Standard Policy')}
Policy Details: {policy_data.get('content', 'Refunds permitted within 30 days of delivery.')}
Auto-Approval Limit: ${policy_data.get('max_automated_limit', 100.0):.2f}
Return Window: {policy_data.get('return_window_days', 30)} days

[Specific Order Lookup Result]
{order_lookup_info}
"""

    response_text = None

    # 1. Attempt LLM Generation with Context Evidence
    if llm:
        try:
            messages = [
                SystemMessage(content=AUTONOMOUS_SYSTEM_PROMPT),
                SystemMessage(content=f"REAL-TIME EVIDENCE CONTEXT:\n{evidence_context}"),
            ] + list(state["messages"][-4:])
            ai_msg = await llm.ainvoke(messages)
            response_text = extract_text_from_message(ai_msg)
        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}. Attempting fallback model...")
            if fallback_llm:
                try:
                    fallback_msg = await fallback_llm.ainvoke(messages)
                    response_text = extract_text_from_message(fallback_msg)
                except Exception:
                    pass

    # 2. Context-Aware Resilient Fallback (Zero Generic Wall-of-Text Dumping)
    if not response_text:
        last_lower = last_msg.lower()
        target_order = extract_order_id_from_context(list(state.get("messages", [])))

        # Format orders summary for reference
        orders_display = "\n".join([
            f"- Order {o['order_id']}: ${o['total']:.2f} ({o['status']}) | Tracking: {o.get('tracking_number', 'N/A')}"
            for o in orders_list
        ]) if orders_list else "No active orders found on file."

        # Scenario 1: Order Cancellation Request
        if any(k in last_lower for k in ["cancel"]):
            if target_order:
                if order_data and order_data.get("found"):
                    current_status = order_data.get("status", "").upper()
                    if current_status == "PROCESSING":
                        cancel_res = await execute_order_cancellation(order_id=target_order, customer_id=state["customer_id"])
                        if cancel_res.get("success"):
                            response_text = (
                                f"Order {target_order} was in PROCESSING status and has been successfully canceled. "
                                f"Because it had not yet shipped, your payment of ${order_data.get('order_total', 0.0):.2f} "
                                f"has been voided and refunded to your original payment method."
                            )
                        else:
                            response_text = cancel_res.get("reason")
                    elif current_status in ["CANCELED", "CANCELLED"]:
                        response_text = f"Order {target_order} has already been canceled. No further cancellation action is needed."
                    elif current_status == "REFUNDED":
                        response_text = f"Order {target_order} has already been refunded in full."
                    elif current_status == "SHIPPED":
                        response_text = (
                            f"Order {target_order} has already shipped (Tracking: {order_data.get('tracking_number', 'N/A')}). "
                            f"Orders in transit cannot be intercepted or canceled while en route. "
                            f"Once your package arrives, you may initiate a return within {policy_data.get('return_window_days', 30)} days."
                        )
                    elif current_status == "DELIVERED":
                        deliv_str = f" on {order_data.get('delivered_date')}" if order_data.get('delivered_date') else ""
                        response_text = (
                            f"Order {target_order} was delivered{deliv_str} and can no longer be canceled directly. "
                            f"If you wish to return your items, you may initiate a return or refund claim under our standard store policy."
                        )
                    else:
                        response_text = f"Order {target_order} is currently in {current_status} status and cannot be canceled directly."
                else:
                    response_text = f"We could not locate order '{target_order}' under your account ({cust_data.get('name', 'Customer')}).\n\nYour active orders on file are:\n{orders_display}\n\nPlease verify your order number so I can assist you."
            else:
                response_text = f"To cancel an order, please specify your Order ID. Orders can be canceled immediately while in PROCESSING status before shipment.\n\nYour recent orders:\n{orders_display}"

        # Scenario 2: Order Tracking / Status Request
        elif any(k in last_lower for k in ["track", "where is", "delivery status", "package", "tracking"]):
            if target_order and order_data and order_data.get("found"):
                delivered_val = order_data.get('delivered_date') or "In Transit"
                tracking_val = order_data.get('tracking_number') or "N/A"
                response_text = f"Order {order_data['order_id']} is currently {order_data['status']}. Total: ${order_data.get('order_total', 0.0):.2f}. Tracking Number: {tracking_val}. Delivery Status: {delivered_val}."
            elif target_order:
                response_text = f"We could not locate order '{target_order}' under your account.\n\nYour registered orders are:\n{orders_display}"
            elif orders_list:
                latest = orders_list[0]
                response_text = f"Here is the status of your recent orders:\n\n{orders_display}\n\nYour most recent order {latest['order_id']} is {latest['status']} with tracking number {latest.get('tracking_number', 'N/A')}."
            else:
                response_text = "Could you please specify which Order ID you would like to track?"

        # Scenario 3: Order Listing / Purchases History
        elif any(k in last_lower for k in ["list", "my orders", "all orders", "show orders", "purchases", "bought", "my purchases"]):
            if orders_list:
                response_text = f"Here are the orders associated with your account ({cust_data.get('name', 'Customer')}):\n\n{orders_display}\n\nPlease let me know if you would like to track, return, or cancel any of these purchases."
            else:
                response_text = "We did not find any recent orders associated with your account. Please check the account number or provide an Order ID."

        # Scenario 4: Refund / Return Inquiry
        elif any(k in last_lower for k in ["refund", "return", "broken", "damaged", "money back", "defective"]):
            if target_order and order_data and not order_data.get("found"):
                response_text = f"We could not locate order '{target_order}' under your account. Please check your order ID from your orders list:\n\n{orders_display}"
            elif target_order and order_data and not order_data.get("is_return_eligible"):
                response_text = f"Order {target_order} was delivered outside our {policy_data.get('return_window_days', 30)}-day return eligibility window. Under store policy, automated refunds cannot be processed for expired orders."
            elif orders_list:
                orders_preview = "\n".join([f"- {o['order_id']} (${o['total']:.2f}, {o['status']})" for o in orders_list[:3]])
                response_text = (
                    f"I can help you with your return or refund request. To make sure I assist with the correct purchase, "
                    f"could you please specify which order you would like to resolve?\n\nHere are your recent orders:\n{orders_preview}\n\n"
                    f"Please reply with the Order ID (e.g. {orders_list[0]['order_id']}) and the reason."
                )
            else:
                response_text = "I can assist you with your refund inquiry. Could you please provide your Order ID and a description of the issue?"

        # Scenario 5: Policy Inquiries (Returns, Warranty, Shipping)
        elif any(k in last_lower for k in ["warranty"]):
            response_text = "All electronics and hardware products include a 1-year limited manufacturer warranty covering repairs or replacements for hardware defects."
        elif any(k in last_lower for k in ["shipping", "transit", "ship"]):
            response_text = "Orders typically arrive within 3-5 business days. If a shipment shows no tracking progression for more than 7 business days, our team can initiate a carrier trace or replacement."
        elif any(k in last_lower for k in ["policy", "return window", "restocking"]):
            response_text = f"Our standard return policy allows items to be returned within {policy_data.get('return_window_days', 30)} days of delivery for damaged or defective goods. Unopened items in original packaging qualify for return within 14 days."

        # Scenario 6: Greetings & General Assistance
        else:
            response_text = f"Hello {cust_data.get('name', 'there')}, I am your ResolveOps support assistant. I can help you view your orders, track shipments, cancel processing orders, or process returns and refunds. How can I help you today?"

    return {"messages": [AIMessage(content=response_text)]}


# ---------------------------------------------------------------------------
# Node 6: Resolution Planner (When refunding a verified eligible order)
# ---------------------------------------------------------------------------
async def resolution_planner_node(state: ResolveOpsState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content if state.get("messages") else ""
    order_id = extract_order_id_from_context(list(state.get("messages", [])))

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
# Routing Guard: Dynamic Threshold from PostgreSQL Policy
# ---------------------------------------------------------------------------
def check_escalation_guard(state: ResolveOpsState) -> Literal["suspend_for_hitl", "execute_mutation"]:
    action = state.get("pending_action")
    if not action:
        return "execute_mutation"

    evidence_map = {e["source"]: e["data"] for e in state.get("investigation_evidence", [])}
    policy_data = evidence_map.get("search_refund_policy", {})
    policy_limit = float(policy_data.get("max_automated_limit", settings.MAX_AUTO_REFUND_LIMIT_DEFAULT))

    if action.action_type == "refund_order" and action.amount > policy_limit:
        return "suspend_for_hitl"

    if state.get("is_escalated", False):
        return "suspend_for_hitl"

    return "execute_mutation"


# ---------------------------------------------------------------------------
# Node 7: Human-in-the-Loop Suspension
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
# Node 8: Action Executor (PostgreSQL Mutation + Redis Locks)
# ---------------------------------------------------------------------------
async def tool_execution_node(state: ResolveOpsState) -> Dict[str, Any]:
    if state.get("admin_approval_status") == "REJECTED":
        return {
            "action_results": [{"status": "ABORTED", "message": "Transaction rejected by supervisor."}],
            "messages": [AIMessage(content="Your refund request was reviewed by a supervisor and declined in accordance with store policy.")]
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
                "messages": [AIMessage(content=f"Successfully processed refund of ${action.amount:.2f} for order {action.entity_id}. Settlement Transaction ID: {res.get('transaction_id')}.")]
            }
        except LockAcquisitionError:
            return {"error_trace": "Resource contention lock held."}
        except ActionInProgressError:
            return {"error_trace": "Duplicate in-flight execution."}
        except Exception as exc:
            logger.error(f"Tool execution failed: {exc}")
            return {
                "action_results": [{"status": "FAILED", "error": str(exc)}],
                "messages": [AIMessage(content=f"An error occurred while executing the transaction: {str(exc)}")]
            }


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
    workflow.add_node("conversational_response", conversational_response_node)
    workflow.add_node("resolution_planner", resolution_planner_node)
    workflow.add_node("hitl_suspension", hitl_suspension_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("verifier_and_memory", verifier_and_memory_node)

    # Main DAG transitions
    workflow.add_edge(START, "guardrail_input")
    workflow.add_edge("guardrail_input", "triage")
    workflow.add_edge("triage", "context_loader")
    workflow.add_edge("context_loader", "investigation")

    # Conditional Branch: Action Mutation vs. Conversational Reasoning
    workflow.add_conditional_edges(
        "investigation",
        check_investigation_sufficiency,
        {
            "conversational_response": "conversational_response",
            "propose_resolution": "resolution_planner",
            "security_abort": END
        }
    )

    # Resolution planner routes to HITL suspension or immediate mutation execution
    workflow.add_conditional_edges(
        "resolution_planner",
        check_escalation_guard,
        {
            "suspend_for_hitl": "hitl_suspension",
            "execute_mutation": "tool_execution"
        }
    )

    workflow.add_edge("conversational_response", "verifier_and_memory")
    workflow.add_edge("hitl_suspension", "tool_execution")
    workflow.add_edge("tool_execution", "verifier_and_memory")
    workflow.add_edge("verifier_and_memory", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
