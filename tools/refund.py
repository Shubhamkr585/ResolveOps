import asyncio
import logging
from core.decorators import transactional_tool

logger = logging.getLogger("resolveops.tools")


@transactional_tool(
    entity_key_param="order_id",
    action_id_param="action_id",
    lock_ttl_seconds=8,
    lock_retry_times=3
)
async def process_refund_tool(
    order_id: str,
    action_id: str,
    amount: float,
    reason: str
) -> dict:
    """
    Executes transaction with payment gateway. Protected by Redis locks and idempotency keys.
    """
    logger.info(f"Processing refund: Order={order_id}, Amount={amount}, Action={action_id}")
    await asyncio.sleep(0.3)  # Gateway network call simulation

    return {
        "transaction_id": f"tx_rf_{order_id}",
        "order_id": order_id,
        "refunded_amount": amount,
        "status": "SETTLED"
    }