import logging
from typing import Dict, Any
from core.db import query_policy_from_db

logger = logging.getLogger("resolveops.tools.policy")


async def search_refund_policy(query: str) -> Dict[str, Any]:
    """
    Retrieves verified policy documents from PostgreSQL using Hybrid Search (Handbook Section 11.2).
    Combines Sparse Lexical FTS (ts_rank) and Dense Semantic Intent Matching via Reciprocal
    Rank Fusion (RRF with k=60) to ground agent reasoning in official policy text.
    """
    logger.info(f"Executing Hybrid Policy Search (FTS + Dense RRF) for: {query[:50]}")
    return await query_policy_from_db(query)
