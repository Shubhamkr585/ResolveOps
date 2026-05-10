import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph

from api.schemas import AdminApprovalWebhookPayload, WebhookResponse
from api.security import verify_admin_signature
from api.dashboard_routes import remove_suspended_interrupt

logger = logging.getLogger("resolveops.api")
router = APIRouter(prefix="/v1/admin/webhooks", tags=["HITL Admin"])


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.resolveops_graph


@router.post(
    "/resume-action",
    response_model=WebhookResponse,
    dependencies=[Depends(verify_admin_signature)]
)
async def resume_suspended_action(
    payload: AdminApprovalWebhookPayload,
    graph: CompiledStateGraph = Depends(get_graph)
) -> WebhookResponse:
    thread_config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        snapshot = await graph.aget_state(thread_config)
    except Exception as exc:
        logger.error(f"Failed to fetch state for thread {payload.thread_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving state from checkpointer."
        )

    if not snapshot or not snapshot.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread '{payload.thread_id}' not found."
        )

    if not snapshot.next:
        return WebhookResponse(
            status="NOOP",
            thread_id=payload.thread_id,
            action_id=payload.action_id,
            detail="Thread is completed. No pending interruption."
        )

    pending_action = snapshot.values.get("pending_action")
    if not pending_action or pending_action.action_id != payload.action_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Action ID mismatch with pending interrupt."
        )

    resume_payload = {
        "status": payload.decision,
        "admin_id": payload.admin_id,
        "override_reason": payload.override_reason
    }

    try:
        resumed_state = await graph.ainvoke(Command(resume=resume_payload), config=thread_config)
        remove_suspended_interrupt(payload.thread_id)
        action_results = resumed_state.get("action_results", [])
        latest_res = action_results[-1] if action_results else None
        final_status = "SUCCESS" if payload.decision == "APPROVED" else "REJECTED_BY_ADMIN"

        return WebhookResponse(
            status=final_status,
            thread_id=payload.thread_id,
            action_id=payload.action_id,
            execution_result=latest_res,
            detail=f"Resumed and executed with status {payload.decision}."
        )
    except Exception as exc:
        logger.exception(f"Error resuming graph: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Downstream failure: {str(exc)}"
        )