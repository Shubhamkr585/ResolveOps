import logging
from typing import Optional, Dict, Any
from core.decorators import transactional_tool
from core.db import execute_order_refund

logger = logging.getLogger("resolveops.tools.refund")


@transactional_tool(
    entity_key_param="order_id",
    action_id_param="action_id",
    lock_ttl_seconds=10,
    lock_retry_times=3
)
async def process_refund_tool(
    order_id: str,
    action_id: str,
    amount: float,
    reason: str,
    customer_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes refund mutation directly against PostgreSQL.
    Protected by Redis distributed mutex (entity lock) and deterministic idempotency token.
    Updates order status and persists audit record in refund_transactions.
    """
    logger.info(f"Executing database refund: Order={order_id}, Amount={amount}, Action={action_id}")
    return await execute_order_refund(
        order_id=order_id,
        action_id=action_id,
        amount=amount,
        reason=reason,
        customer_id=customer_id
    )
