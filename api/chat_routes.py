import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("resolveops.chat")
router = APIRouter(prefix="/v1/chat", tags=["Customer Chat"])


class ChatMessageRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "I want a full refund of $120 for order ORD-9821. It arrived completely broken."})
    customer_id: str = Field(default="CUST-1001")
    thread_id: Optional[str] = Field(default=None, description="Provide to continue existing conversation thread")


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "COMPLETED" or "SUSPENDED_FOR_APPROVAL"
    reply: Optional[str] = None
    interruption_details: Optional[Dict[str, Any]] = None


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.resolveops_graph


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    payload: ChatMessageRequest,
    request: Request
) -> ChatResponse:
    graph: CompiledStateGraph = get_graph(request)
    thread_id = payload.thread_id or str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    task_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    initial_input = {
        "messages": [HumanMessage(content=payload.message)],
        "customer_id": payload.customer_id,
        "session_id": thread_id,
        "task_id": task_id,
        "run_id": run_id,
        "investigation_evidence": [],
        "action_results": [],
        "audit_events": [],
        "is_escalated": False
    }

    try:
        # Stream / run graph until completion or interrupt()
        res = await graph.ainvoke(initial_input, config=thread_config)
        state_snapshot = await graph.aget_state(thread_config)

        # Check if execution was halted by LangGraph interrupt()
        if state_snapshot.next:
            interrupt_payload = {}
            if state_snapshot.tasks and hasattr(state_snapshot.tasks[0], "interrupts") and state_snapshot.tasks[0].interrupts:
                interrupt_payload = state_snapshot.tasks[0].interrupts[0].value

            reply_text = (
                interrupt_payload.get("message")
                or f"Your refund request for order {interrupt_payload.get('entity_id', '')} of ${interrupt_payload.get('amount', 0.0):.2f} exceeds automated limits ($100.00). It has been sent to a human supervisor for review."
            )
            return ChatResponse(
                thread_id=thread_id,
                status="SUSPENDED_FOR_APPROVAL",
                reply=reply_text,
                interruption_details=interrupt_payload
            )

        # If completed normally
        last_message = res["messages"][-1].content if (res and res.get("messages")) else "Request processed."
        return ChatResponse(
            thread_id=thread_id,
            status="COMPLETED",
            reply=str(last_message)
        )

    except Exception as exc:
        logger.exception(f"Agent execution failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failure: {str(exc)}"
        )