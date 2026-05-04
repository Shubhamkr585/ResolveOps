import time
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("resolveops.observability")


class AuditLogger:
    """
    Structured observability logger implementing Section 9 of the
    LangGraph Agent Engineering Handbook.
    Produces structured JSON audit trails with correlation IDs.
    """

    @staticmethod
    def log_event(
        event_type: str,
        thread_id: str,
        customer_id: str,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        status: str = "INFO",
        latency_ms: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        event_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "thread_id": thread_id,
            "customer_id": customer_id,
            "run_id": run_id or "run-default",
            "task_id": task_id or "task-default",
            "status": status,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
            "payload": payload or {},
            "error_message": error_message
        }

        # Structured JSON emission
        log_line = json.dumps(event_record)
        if status in ("ERROR", "SECURITY_VIOLATION"):
            logger.error(log_line)
        elif status == "WARNING":
            logger.warning(log_line)
        else:
            logger.info(log_line)

        return event_record


class ExecutionTimer:
    """Context manager to measure node and tool execution latency."""

    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
