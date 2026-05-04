import logging
from typing import Dict, Any
from core.db import query_policy_from_db

logger = logging.getLogger("resolveops.tools.policy")


async def search_refund_policy(query: str) -> Dict[str, Any]:
    """
    Retrieves verified policy documents from PostgreSQL policies table (Handbook Section 11.2).
    Grounds agent reasoning in official policy text rather than model hallucination.
    """
    logger.info(f"Querying policies table in PostgreSQL for: {query[:50]}")
    return await query_policy_from_db(query)
