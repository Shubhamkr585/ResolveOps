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

    @classmethod
    async def apply_context_window_budget(
        cls,
        messages: List[BaseMessage],
        existing_summary: Optional[str] = None,
        llm: Optional[Any] = None
    ) -> Tuple[List[BaseMessage], Optional[str]]:
        """
        Sliding Window + Progressive Semantic Summarization (Handbook Section 5.3, 5.4).
        Preserves the latest N messages verbatim and progressively summarizes older turns using
        semantic LLM extraction to retain critical business entities (Order IDs, amounts, reasons).
        Includes a robust entity-preserving fallback when the model is offline.
        """
        if len(messages) <= MAX_VERBATIM_MESSAGES:
            return messages, existing_summary

        older_messages = messages[:-MAX_VERBATIM_MESSAGES]
        recent_messages = list(messages[-MAX_VERBATIM_MESSAGES:])

        # Build lines of older dialogue to be incorporated
        dialogue_lines = []
        for msg in older_messages:
            role = "Customer" if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human" else "Agent"
            content = msg.content if hasattr(msg, "content") else str(msg)
            # Normalize whitespace
            content_clean = " ".join(str(content).split())
            dialogue_lines.append(f"{role}: {content_clean}")
        dialogue_block = "\n".join(dialogue_lines)

        # 1. Semantic LLM Summarization (Preserves context, entities, agreements)
        if llm:
            try:
                prompt = (
                    "You are the memory summarizer for ResolveOps customer support.\n"
                    "Progressively update the conversation summary with the new conversation turns.\n"
                    "CRITICAL RULES:\n"
                    "1. Preserve all key business entities: exact Order IDs (e.g. ORD-XXXX), dollar amounts ($XX.XX), tracking numbers, and reasons.\n"
                    "2. Capture customer intent and agreed outcomes.\n"
                    "3. Keep the summary concise (under 120 words), factual, and coherent. Do not use emojis.\n\n"
                    f"CURRENT SUMMARY:\n{existing_summary or 'No prior summary.'}\n\n"
                    f"NEW CONVERSATION TURNS TO INCORPORATE:\n{dialogue_block}\n\n"
                    "UPDATED PROGRESSIVE SUMMARY:"
                )
                res = await llm.ainvoke(prompt)
                summary_text = res.content.strip() if hasattr(res, "content") else str(res).strip()
                if summary_text:
                    return recent_messages, summary_text
            except Exception as exc:
                logger.warning(f"Semantic LLM summarization failed: {exc}. Using entity-preserving heuristic.")

        # 2. Entity-Preserving Robust Fallback (Zero crude mid-word chopping)
        summary_parts = []
        if existing_summary:
            summary_parts.append(f"Prior: {existing_summary}")

        for line in dialogue_lines:
            # Preserve first complete sentence or up to 140 chars at word boundary
            if len(line) <= 140:
                summary_parts.append(line)
            else:
                cut = line[:140]
                last_space = cut.rfind(" ")
                clean_cut = cut[:last_space] if last_space > 80 else cut
                summary_parts.append(f"{clean_cut}...")

        compact_summary = " | ".join(summary_parts)
        return recent_messages, compact_summary
