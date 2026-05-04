import operator
from typing import Annotated, Sequence, TypedDict, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class PendingAction(BaseModel):
    action_id: str = Field(description="Deterministic idempotency key (UUIDv5)")
    action_type: Literal["refund_order", "cancel_order", "update_address"]
    entity_id: str = Field(description="Order ID or Payment ID")
    amount: float = Field(default=0.0)
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TriageDecision(BaseModel):
    route: Literal["policy_rag", "order_management", "billing_refund", "human_escalation"]
    sentiment_score: float = Field(description="-1.0 (toxic/furious) to 1.0 (satisfied)")
    requires_escalation: bool
    reason: str


class ResolveOpsState(TypedDict):
    """
    Comprehensive state schema complying with Section 3 and Page 17 of the
    LangGraph Agent Engineering Handbook.
    Separates working memory, evidence, and verified tool mutations.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_id: str
    session_id: str  # thread_id
    task_id: Optional[str]
    run_id: Optional[str]
    summary: Optional[str]  # Rolling compact summary (Section 4.7 & 5.5)
    customer_memory: Optional[Dict[str, Any]]  # Long-term Store profile (Section 4.3)
    current_route: Optional[str]
    triage: Optional[TriageDecision]
    investigation_evidence: Annotated[List[Dict[str, Any]], operator.add]
    pending_action: Optional[PendingAction]
    action_results: Annotated[List[Dict[str, Any]], operator.add]
    is_escalated: bool
    escalation_reason: Optional[str]
    admin_approval_status: Optional[Literal["APPROVED", "REJECTED"]]
    audit_events: Annotated[List[Dict[str, Any]], operator.add]
    error_trace: Optional[str]
