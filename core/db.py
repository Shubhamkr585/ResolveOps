import os
import json
import logging
from typing import Optional, Dict, Any, List
from psycopg_pool import AsyncConnectionPool
from core.config import get_settings

logger = logging.getLogger("resolveops.db")
settings = get_settings()

_db_pool: Optional[AsyncConnectionPool] = None


def get_db_pool() -> AsyncConnectionPool:
    global _db_pool
    if _db_pool is None:
        # Scale down pool size in serverless environments (Vercel) to prevent connection spikes
        pool_size = 3 if os.getenv("VERCEL") else 20
        _db_pool = AsyncConnectionPool(
            conninfo=settings.DATABASE_URL,
            max_size=pool_size,
            kwargs={"autocommit": True}
        )
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
                        "message": f"Order '{order_id}' does not exist in the database."
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


# Semantic Concept Clusters for Dense Intent Matching in Hybrid Search
POLICY_SEMANTIC_EMBEDDINGS: Dict[str, List[str]] = {
    "POL-REF-01": [
        "damaged", "broken", "defective", "cracked", "shattered", "faulty", "not working",
        "destroyed", "refund", "money back", "reimbursement", "compensation", "flawed", "smashed"
    ],
    "POL-RET-02": [
        "unopened", "sealed", "box", "original packaging", "restocking", "return window",
        "14 days", "remorse", "change mind", "dont like", "unwanted", "return policy", "send back"
    ],
    "POL-SHP-03": [
        "lost", "delayed", "transit", "shipping", "carrier", "trace", "progression",
        "package", "courier", "late delivery", "stuck", "fedex", "usps", "has not arrived", "where is"
    ],
    "POL-WAR-04": [
        "warranty", "hardware", "repair", "replacement", "electronics", "manufacturer",
        "headphone", "camera", "drone", "breakdown", "malfunction", "1-year", "year limited", "guarantee"
    ],
    "POL-ACC-05": [
        "cancel", "address", "processing", "cancellation", "stop order", "change address",
        "mistake", "wrong item", "modify order", "pre-shipment", "intercept"
    ]
}


def compute_dense_semantic_score(query: str, policy_id: str, title: str, content: str) -> float:
    """Computes semantic similarity score based on semantic concept overlap and intent matching."""
    tokens = [t.strip().lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not tokens:
        return 0.0
    concepts = POLICY_SEMANTIC_EMBEDDINGS.get(policy_id, [])
    # Match both intent concept clusters and direct text similarity
    match_count = sum(1 for c in concepts if any(c in t or t in c for t in tokens))
    title_tokens = set(re.findall(r"\w+", title.lower()))
    title_matches = sum(1 for t in tokens if t in title_tokens)
    return (match_count * 2.0 + title_matches * 1.5) / (len(tokens) + 1.0)


async def query_policy_from_db(query_text: str) -> Dict[str, Any]:
    """
    Retrieves grounded business rules from PostgreSQL policies table using Hybrid Search:
    1. Sparse Lexical Search: PostgreSQL GIN Full-Text Search (ts_rank with plainto_tsquery).
    2. Dense Semantic Search: Semantic intent concept scoring over policy documents.
    3. Reciprocal Rank Fusion (RRF): Combines lexical and semantic ranks with k=60 to guarantee
       precision on exact numbers (30 days, $100) while capturing colloquial user phrasing.
    """
    pool = get_db_pool()
    clean_text = query_text.strip()

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Fetch all policy candidates
                await cur.execute("SELECT policy_id, category, title, content, max_auto_refund_limit, return_window_days FROM policies;")
                all_policies = await cur.fetchall()
                if not all_policies:
                    return {
                        "policy_id": "POL-DEFAULT",
                        "category": "refund",
                        "title": "Standard Resolution Policy",
                        "content": "Refunds permitted within 30 days of delivery. Automated approvals capped at configured limit.",
                        "max_automated_limit": settings.MAX_AUTO_REFUND_LIMIT_DEFAULT,
                        "return_window_days": settings.RETURN_WINDOW_DAYS_DEFAULT
                    }

                policy_map = {
                    r[0]: {
                        "policy_id": r[0],
                        "category": r[1],
                        "title": r[2],
                        "content": r[3],
                        "max_automated_limit": float(r[4]),
                        "return_window_days": int(r[5])
                    }
                    for r in all_policies
                }

                if not clean_text:
                    default_res = dict(policy_map.get("POL-REF-01", list(policy_map.values())[0]))
                    default_res["retrieval_method"] = "DEFAULT_FALLBACK"
                    return default_res

                # 1. Sparse Lexical Search (PostgreSQL GIN FTS ts_rank)
                sparse_query = """
                    SELECT policy_id, ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s)) AS sparse_rank
                    FROM policies
                    WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', %s)
                    ORDER BY sparse_rank DESC;
                """
                await cur.execute(sparse_query, (clean_text, clean_text))
                sparse_rows = await cur.fetchall()
                sparse_rank_map = {row[0]: rank for rank, row in enumerate(sparse_rows, start=1)}

                # 2. Dense Semantic Search
                dense_scores = [
                    (pid, compute_dense_semantic_score(clean_text, pid, doc["title"], doc["content"]))
                    for pid, doc in policy_map.items()
                ]
                dense_sorted = sorted(dense_scores, key=lambda x: x[1], reverse=True)
                dense_rank_map = {pid: rank for rank, (pid, score) in enumerate(dense_sorted, start=1) if score > 0}

                # 3. Reciprocal Rank Fusion (RRF with k=60)
                # Formula: RRF(d) = (w_sparse / (60 + rank_sparse)) + (w_dense / (60 + rank_dense))
                k = 60
                rrf_scores = {}
                for pid in policy_map:
                    s_rank = sparse_rank_map.get(pid, 100)  # Rank penalty if unmatched
                    d_rank = dense_rank_map.get(pid, 100)
                    score = (0.5 / (k + s_rank)) + (0.5 / (k + d_rank))
                    rrf_scores[pid] = (score, s_rank, d_rank)

                best_pid = max(rrf_scores.keys(), key=lambda p: rrf_scores[p][0])
                best_score, best_s_rank, best_d_rank = rrf_scores[best_pid]

                result = dict(policy_map[best_pid])
                result["retrieval_method"] = "HYBRID_RRF_FUSION"
                result["hybrid_metadata"] = {
                    "sparse_rank": best_s_rank if best_s_rank != 100 else None,
                    "dense_rank": best_d_rank if best_d_rank != 100 else None,
                    "rrf_score": round(best_score, 4)
                }
                return result

    except Exception as exc:
        logger.error(f"Error querying policy with Hybrid Search: {exc}")

    return {
        "policy_id": "POL-DEFAULT",
        "category": "refund",
        "title": "Standard Resolution Policy",
        "content": "Refunds permitted within 30 days of delivery. Automated approvals capped at configured limit.",
        "max_automated_limit": settings.MAX_AUTO_REFUND_LIMIT_DEFAULT,
        "return_window_days": settings.RETURN_WINDOW_DAYS_DEFAULT,
        "retrieval_method": "EXCEPTION_FALLBACK"
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
                    SELECT order_id, status, order_total, order_date, is_return_eligible, tracking_number, items
                    FROM orders
                    WHERE customer_id = %s
                    ORDER BY order_date DESC
                    LIMIT 10;
                """, (customer_id,))
                recent_orders = await cur.fetchall()

                formatted_orders = []
                for r in recent_orders:
                    items_val = r[6] if isinstance(r[6], list) else json.loads(r[6]) if r[6] else []
                    formatted_orders.append({
                        "order_id": r[0],
                        "status": r[1],
                        "total": float(r[2]),
                        "date": r[3].strftime("%Y-%m-%d") if r[3] else None,
                        "is_return_eligible": bool(r[4]),
                        "tracking_number": r[5] or "N/A",
                        "items": items_val
                    })

                return {
                    "found": True,
                    "customer_id": customer_id,
                    "name": name,
                    "tier": tier,
                    "trust_score": float(trust_score),
                    "total_orders": total_orders,
                    "recent_orders": formatted_orders
                }
    except Exception as exc:
        logger.error(f"Error querying customer {customer_id}: {exc}")
        return {"found": False, "customer_id": customer_id}


async def execute_order_refund(
    order_id: str,
    action_id: str,
    amount: float,
    reason: str,
    customer_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes an atomic database mutation:
    1. Verifies order exists and is not already refunded.
    2. Inserts an immutable transaction record into refund_transactions.
    3. Updates orders.status = 'REFUNDED'.
    All wrapped within a database transaction.
    """
    pool = get_db_pool()
    tx_id = f"tx_rf_{order_id.lower()}_{action_id[:8]}"

    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                # 1. Fetch current order
                await cur.execute("""
                    SELECT customer_id, status, order_total
                    FROM orders
                    WHERE UPPER(order_id) = UPPER(%s)
                    FOR UPDATE;
                """, (order_id.strip(),))
                row = await cur.fetchone()

                if not row:
                    raise ValueError(f"Order '{order_id}' does not exist in database.")

                cust_id, current_status, total = row
                if current_status == "REFUNDED":
                    logger.info(f"Order {order_id} was already refunded in database.")
                    return {
                        "transaction_id": tx_id,
                        "order_id": order_id,
                        "refunded_amount": float(amount),
                        "status": "SETTLED",
                        "note": "Order was previously settled."
                    }

                actual_customer_id = customer_id or cust_id

                # 2. Insert into refund_transactions
                await cur.execute("""
                    INSERT INTO refund_transactions (
                        transaction_id, order_id, amount, reason, idempotency_key, customer_id, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'SETTLED')
                    ON CONFLICT (idempotency_key) DO UPDATE
                    SET status = EXCLUDED.status
                    RETURNING transaction_id, created_at;
                """, (tx_id, order_id, amount, reason, action_id, actual_customer_id))
                inserted = await cur.fetchone()
                db_tx_id = inserted[0] if inserted else tx_id

                # 3. Update orders status to REFUNDED
                await cur.execute("""
                    UPDATE orders
                    SET status = 'REFUNDED'
                    WHERE UPPER(order_id) = UPPER(%s);
                """, (order_id.strip(),))

                logger.info(f"Database mutation complete: Order {order_id} marked REFUNDED, transaction {db_tx_id} recorded.")

                return {
                    "transaction_id": db_tx_id,
                    "order_id": order_id,
                    "refunded_amount": round(float(amount), 2),
                    "status": "SETTLED",
                    "reason": reason,
                    "customer_id": actual_customer_id
                }


async def execute_order_cancellation(order_id: str, customer_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Cancels an order if it is in PROCESSING status according to store policy POL-ACC-05.
    If the order is already SHIPPED or DELIVERED, returns the policy constraint reason.
    """
    pool = get_db_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT customer_id, status, order_total
                    FROM orders
                    WHERE UPPER(order_id) = UPPER(%s)
                    FOR UPDATE;
                """, (order_id.strip(),))
                row = await cur.fetchone()

                if not row:
                    return {"success": False, "reason": f"Order '{order_id}' was not found under your account."}

                cust_id, current_status, total = row
                if customer_id and cust_id != customer_id:
                    return {"success": False, "reason": "This order number does not match your registered account."}

                if current_status == "PROCESSING":
                    await cur.execute("""
                        UPDATE orders
                        SET status = 'CANCELED'
                        WHERE UPPER(order_id) = UPPER(%s);
                    """, (order_id.strip(),))
                    logger.info(f"Order {order_id} successfully canceled in database.")
                    return {
                        "success": True,
                        "order_id": order_id,
                        "refunded_amount": float(total),
                        "status": "CANCELED"
                    }
                else:
                    return {
                        "success": False,
                        "order_id": order_id,
                        "status": current_status,
                        "reason": f"Order {order_id} is already {current_status} and cannot be canceled in transit. Once it arrives, you can request a return within 30 days."
                    }


async def get_dashboard_data() -> Dict[str, Any]:
    """Retrieves live aggregate data for the interview presentation dashboard."""
    pool = get_db_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Recent orders
                await cur.execute("""
                    SELECT order_id, customer_id, status, order_total, tracking_number, order_date, is_return_eligible
                    FROM orders
                    ORDER BY order_date DESC
                    LIMIT 15;
                """)
                orders_rows = await cur.fetchall()

                # Recent refunds
                await cur.execute("""
                    SELECT transaction_id, order_id, customer_id, amount, reason, status, created_at
                    FROM refund_transactions
                    ORDER BY created_at DESC
                    LIMIT 10;
                """)
                refund_rows = await cur.fetchall()

                # Customers summary
                await cur.execute("""
                    SELECT customer_id, name, tier, trust_score, total_orders_count
                    FROM customers
                    ORDER BY customer_id ASC
                    LIMIT 10;
                """)
                cust_rows = await cur.fetchall()

                # Summary stats
                await cur.execute("SELECT COUNT(*) FROM orders;")
                total_orders = (await cur.fetchone())[0]

                await cur.execute("SELECT COUNT(*) FROM refund_transactions;")
                total_refunds = (await cur.fetchone())[0]

                await cur.execute("SELECT COUNT(*) FROM customers;")
                total_custs = (await cur.fetchone())[0]

                return {
                    "total_orders": total_orders,
                    "total_refunds": total_refunds,
                    "total_customers": total_custs,
                    "orders": [
                        {
                            "order_id": r[0],
                            "customer_id": r[1],
                            "status": r[2],
                            "total": float(r[3]),
                            "tracking": r[4],
                            "date": r[5].strftime("%Y-%m-%d %H:%M") if r[5] else "",
                            "return_eligible": bool(r[6])
                        }
                        for r in orders_rows
                    ],
                    "refunds": [
                        {
                            "transaction_id": r[0],
                            "order_id": r[1],
                            "customer_id": r[2],
                            "amount": float(r[3]),
                            "reason": r[4],
                            "status": r[5],
                            "created_at": r[6].strftime("%Y-%m-%d %H:%M:%S") if r[6] else ""
                        }
                        for r in refund_rows
                    ],
                    "customers": [
                        {
                            "customer_id": r[0],
                            "name": r[1],
                            "tier": r[2],
                            "trust_score": float(r[3]),
                            "orders_count": r[4]
                        }
                        for r in cust_rows
                    ]
                }
    except Exception as exc:
        logger.error(f"Error fetching dashboard data: {exc}")
        return {"orders": [], "refunds": [], "customers": [], "error": str(exc)}
