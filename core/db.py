import os
import json
import logging
from typing import Optional, Dict, Any, List
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

logger = logging.getLogger("resolveops.db")
load_dotenv()

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5433/resolveops")

_db_pool: Optional[AsyncConnectionPool] = None


def get_db_pool() -> AsyncConnectionPool:
    global _db_pool
    if _db_pool is None:
        _db_pool = AsyncConnectionPool(conninfo=DB_URI, max_size=15, kwargs={"autocommit": True})
    return _db_pool


async def query_order_from_db(order_id: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes real database lookup on the PostgreSQL orders table (Handbook Section 6.2).
    Enforces customer isolation: prevents customers from reading other customers' orders.
    """
    pool = get_db_pool()
    query = """
        SELECT o.order_id, o.customer_id, o.status, o.order_total, o.items,
               o.order_date, o.delivered_date, o.is_return_eligible, o.tracking_number,
               c.name as customer_name, c.tier as customer_tier, c.trust_score
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE UPPER(o.order_id) = UPPER(%s);
    """

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (order_id.strip(),))
                row = await cur.fetchone()

                if not row:
                    return {
                        "found": False,
                        "order_id": order_id,
                        "message": f"Order '{order_id}' does not exist in PostgreSQL database."
                    }

                ord_id, cust_id, status, total, items_raw, ord_date, deliv_date, return_eligible, trk, c_name, c_tier, c_trust = row

                # Customer Authorization Guardrail (Handbook Section 10.2)
                if customer_id and cust_id != customer_id:
                    logger.warning(f"Tenant Guard: {customer_id} denied access to order {order_id} belonging to {cust_id}")
                    return {
                        "found": False,
                        "order_id": order_id,
                        "message": "Authorization Denied: This order belongs to a different customer account."
                    }

                items_list = items_raw if isinstance(items_raw, list) else json.loads(items_raw) if items_raw else []

                return {
                    "found": True,
                    "order_id": ord_id,
                    "customer_id": cust_id,
                    "customer_name": c_name,
                    "customer_tier": c_tier,
                    "customer_trust_score": float(c_trust),
                    "status": status,
                    "order_total": float(total),
                    "items": items_list,
                    "is_return_eligible": bool(return_eligible),
                    "tracking_number": trk,
                    "delivered_date": deliv_date.isoformat() if deliv_date else None
                }
    except Exception as exc:
        logger.error(f"Error querying PostgreSQL for order {order_id}: {exc}")
        return {"found": False, "order_id": order_id, "error": str(exc)}


async def query_policy_from_db(query_text: str) -> Dict[str, Any]:
    """
    Retrieves grounded business rules from PostgreSQL policies table (Handbook Section 11.2).
    """
    pool = get_db_pool()
    category = "refund"
    q_lower = query_text.lower()
    if any(k in q_lower for k in ["ship", "track", "delay", "lost"]):
        category = "shipping"
    elif any(k in q_lower for k in ["cancel", "address", "modify"]):
        category = "cancellation"
    elif any(k in q_lower for k in ["unopened", "mind", "change"]):
        category = "return"
    elif any(k in q_lower for k in ["warranty", "repair", "year"]):
        category = "warranty"

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT policy_id, title, content, max_auto_refund_limit, return_window_days
                    FROM policies
                    WHERE category = %s
                    LIMIT 1;
                """, (category,))
                row = await cur.fetchone()

                if row:
                    p_id, title, content, max_limit, window_days = row
                    return {
                        "policy_id": p_id,
                        "title": title,
                        "content": content,
                        "max_automated_limit": float(max_limit),
                        "return_window_days": int(window_days)
                    }
    except Exception as exc:
        logger.error(f"Error querying policy: {exc}")

    return {
        "policy_id": "POL-FALLBACK",
        "title": "General Policy",
        "content": "Refunds permitted within 30 days of delivery. Limits capped at $100.00 for automation.",
        "max_automated_limit": 100.0,
        "return_window_days": 30
    }


async def query_customer_history_from_db(customer_id: str) -> Dict[str, Any]:
    """
    Retrieves customer order history and trust metrics from PostgreSQL.
    """
    pool = get_db_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT name, email, tier, trust_score, total_orders_count
                    FROM customers
                    WHERE customer_id = %s;
                """, (customer_id,))
                cust = await cur.fetchone()

                if not cust:
                    return {"found": False, "customer_id": customer_id}

                name, email, tier, trust_score, total_orders = cust

                await cur.execute("""
                    SELECT order_id, status, order_total, order_date
                    FROM orders
                    WHERE customer_id = %s
                    ORDER BY order_date DESC
                    LIMIT 5;
                """, (customer_id,))
                recent_orders = await cur.fetchall()

                return {
                    "found": True,
                    "customer_id": customer_id,
                    "name": name,
                    "tier": tier,
                    "trust_score": float(trust_score),
                    "total_orders": total_orders,
                    "recent_orders": [
                        {"order_id": r[0], "status": r[1], "total": float(r[2]), "date": r[3].isoformat() if r[3] else None}
                        for r in recent_orders
                    ]
                }
    except Exception as exc:
        logger.error(f"Error querying customer {customer_id}: {exc}")
        return {"found": False, "customer_id": customer_id}
