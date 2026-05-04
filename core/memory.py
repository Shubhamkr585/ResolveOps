import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.store.base import BaseStore

logger = logging.getLogger("resolveops.memory")

# Sliding Window Configuration (Handbook Section 5.4)
MAX_VERBATIM_MESSAGES = 6


class MemoryManager:
    """
    Implements Memory Architecture & Context Engineering according to
    Sections 4 and 5 of the LangGraph Agent Engineering Handbook.
    
    Scopes:
    1. Working Memory: Graph state variables within a single run.
    2. Short-Term Memory: Checkpointed message turns within a thread.
    3. Long-Term Memory: Cross-thread persistent Store scoped by customer.
    """

    @staticmethod
    def get_customer_namespace(customer_id: str) -> Tuple[str, str]:
        """Scoped namespace isolating customer long-term records (Section 4.3)."""
        return ("customers", customer_id)

    @classmethod
    async def retrieve_long_term_memory(
        cls,
        store: Optional[BaseStore],
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Retrieves durable customer preferences, account tier, and historical disputes (Section 4.6).
        """
        if not store:
            return {}

        namespace = cls.get_customer_namespace(customer_id)
        try:
            profile_item = await store.aget(namespace, "profile")
            disputes_item = await store.aget(namespace, "dispute_summary")

            return {
                "profile": profile_item.value if profile_item else {
                    "tier": "Standard",
                    "preferred_resolution": "refund_to_original_payment",
                    "trust_score": 0.95
                },
                "dispute_history": disputes_item.value if disputes_item else {
                    "total_refunds_count": 0,
                    "last_refund_date": None
                }
            }
        except Exception as exc:
            logger.warning(f"Error retrieving long-term memory for {customer_id}: {exc}")
            return {}

    @classmethod
    async def persist_long_term_fact(
        cls,
        store: Optional[BaseStore],
        customer_id: str,
        fact_key: str,
        fact_value: Dict[str, Any]
    ) -> bool:
        """
        Persists a verified, durable fact under customer namespace with provenance (Section 4.5).
        """
        if not store:
            return False

        namespace = cls.get_customer_namespace(customer_id)
        fact_record = {
            **fact_value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "verified_agent_execution",
            "confidence": 1.0
        }

        try:
            await store.aput(namespace, key=fact_key, value=fact_record)
            logger.info(f"Persisted long-term memory fact '{fact_key}' for customer {customer_id}")
            return True
        except Exception as exc:
            logger.error(f"Failed to persist long-term memory: {exc}")
            return False

    @staticmethod
    def apply_context_window_budget(
        messages: List[BaseMessage],
        existing_summary: Optional[str] = None
    ) -> Tuple[List[BaseMessage], Optional[str]]:
        """
        Sliding Window + Rolling Summary Strategy (Handbook Section 5.3, 5.4).
        Preserves the latest N messages verbatim and rolls older messages into a compact summary.
        """
        if len(messages) <= MAX_VERBATIM_MESSAGES:
            return messages, existing_summary

        older_messages = messages[:-MAX_VERBATIM_MESSAGES]
        recent_messages = list(messages[-MAX_VERBATIM_MESSAGES:])

        # Build compact rolling summary of older messages
        summary_lines = []
        if existing_summary:
            summary_lines.append(f"Previous Context: {existing_summary}")

        for msg in older_messages:
            role = "Customer" if isinstance(msg, HumanMessage) else "Agent"
            content_preview = msg.content[:120].strip()
            summary_lines.append(f"{role}: {content_preview}")

        compact_summary = " | ".join(summary_lines)
        # Cap summary token size
        if len(compact_summary) > 600:
            compact_summary = compact_summary[-600:]

        return recent_messages, compact_summary
