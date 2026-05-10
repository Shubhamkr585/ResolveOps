import os
import hmac
import hashlib
import json
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph

from core.config import get_settings
from core.db import get_dashboard_data, get_db_pool

logger = logging.getLogger("resolveops.dashboard")
settings = get_settings()
router = APIRouter(tags=["Presentation Dashboard"])

# In-memory tracking of recent suspended interrupts for quick dashboard display
_active_interrupts: Dict[str, Dict[str, Any]] = {}


def register_suspended_interrupt(thread_id: str, interruption_details: Dict[str, Any]) -> None:
    """Stores active interruption details for supervisor visibility."""
    if thread_id and interruption_details:
        _active_interrupts[thread_id] = {
            "thread_id": thread_id,
            **interruption_details
        }


def remove_suspended_interrupt(thread_id: str) -> None:
    """Removes resolved interruption from tracking."""
    _active_interrupts.pop(thread_id, None)


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the interactive single-page interview demonstration dashboard."""
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "dashboard.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>ResolveOps Dashboard Template Missing</h1>", status_code=500)


@router.get("/api/dashboard/data")
async def api_dashboard_data():
    """Returns real-time orders, refunds, and customer accounts from PostgreSQL."""
    return await get_dashboard_data()


@router.get("/api/dashboard/active-interrupts")
async def api_active_interrupts(request: Request):
    """Returns active threads awaiting supervisor approval."""
    graph: CompiledStateGraph = request.app.state.resolveops_graph
    results = []

    # Check registered active interrupts
    for thread_id, details in list(_active_interrupts.items()):
        try:
            snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
            if snapshot and snapshot.next:
                results.append(details)
            else:
                # State already completed or resumed
                _active_interrupts.pop(thread_id, None)
        except Exception:
            _active_interrupts.pop(thread_id, None)

    return results


class DashboardResumeRequest(BaseModel):
    thread_id: str
    action_id: str
    decision: str  # "APPROVED" or "REJECTED"
    admin_id: str = "SUPERVISOR-DASHBOARD"
    override_reason: str = "Decision made via presentation dashboard"


@router.post("/api/dashboard/resume-action")
async def api_dashboard_resume(payload: DashboardResumeRequest, request: Request):
    """
    Convenience endpoint for the UI dashboard.
    Automatically signs the payload with HMAC-SHA256 and resumes the LangGraph thread.
    """
    graph: CompiledStateGraph = request.app.state.resolveops_graph
    thread_config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        snapshot = await graph.aget_state(thread_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error fetching state: {exc}")

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread not found.")

    if not snapshot.next:
        remove_suspended_interrupt(payload.thread_id)
        return {"status": "NOOP", "detail": "Thread already completed."}

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

        return {
            "status": "SUCCESS" if payload.decision == "APPROVED" else "REJECTED",
            "thread_id": payload.thread_id,
            "action_id": payload.action_id,
            "execution_result": latest_res,
            "detail": f"Action {payload.decision} successfully processed."
        }
    except Exception as exc:
        logger.exception(f"Error resuming graph via dashboard: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
