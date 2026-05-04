import logging
from typing import Dict, Any, Optional
from core.db import query_order_from_db, query_customer_history_from_db

logger = logging.getLogger("resolveops.tools.order")


async def get_order_details(order_id: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Read-only order lookup tool executing against real PostgreSQL orders table.
    Enforces authorization check inside the tool rather than relying solely on prompts.
    """
    logger.info(f"Looking up order '{order_id}' in PostgreSQL for customer '{customer_id}'")
    return await query_order_from_db(order_id=order_id, customer_id=customer_id)


async def get_customer_history(customer_id: str) -> Dict[str, Any]:
    """
    Read-only customer lookup tool querying customer profile and past orders from PostgreSQL.
    """
    logger.info(f"Looking up customer history for '{customer_id}' in PostgreSQL")
    return await query_customer_history_from_db(customer_id=customer_id)
