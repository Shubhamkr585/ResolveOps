from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class AdminApprovalWebhookPayload(BaseModel):
    thread_id: str = Field(..., description="Unique LangGraph thread identifier")
    action_id: str = Field(..., description="Deterministic UUIDv5 idempotency key")
    decision: Literal["APPROVED", "REJECTED"] = Field(..., description="Supervisor verdict")
    admin_id: str = Field(..., description="Reviewing supervisor ID")
    override_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    status: Literal["SUCCESS", "REJECTED_BY_ADMIN", "NOOP", "ERROR"]
    thread_id: str
    action_id: str
    execution_result: Optional[Dict[str, Any]] = None
    detail: str