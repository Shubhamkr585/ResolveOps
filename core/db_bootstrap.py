import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("resolveops.bootstrap")

# Canonical Policy Knowledge Base
INITIAL_POLICIES = [
    {
        "policy_id": "POL-REF-01",
        "category": "refund",
        "title": "General Defective & Damaged Goods Refund Policy",
        "content": (
            "Customers are entitled to a full or partial refund for items that arrive damaged, defective, or not as described. "
            "Claims must be lodged within 30 days of confirmed delivery. For items damaged in transit, photographic evidence "
            "or inspection is required. Automated refunds are capped at $100.00. Any refund above $100.00 must be escalated "
            "to a human supervisor for manual verification and sign-off."
        ),
        "max_auto_refund_limit": 100.00,
        "return_window_days": 30
    },
    {
        "policy_id": "POL-RET-02",
        "category": "return",
        "title": "Unopened & Change of Mind Returns",
        "content": (
            "Unopened items in original packaging may be returned within 14 days of delivery for a refund minus a 10% restocking fee. "
            "Return shipping costs are the customer's responsibility unless the customer has Gold or VIP_Platinum tier status, "
            "in which case return shipping is waived."
        ),
        "max_auto_refund_limit": 100.00,
        "return_window_days": 14
    },
    {
        "policy_id": "POL-SHP-03",
        "category": "shipping",
        "title": "Lost or Severely Delayed In-Transit Orders",
        "content": (
            "If a package has not shown tracking progression for more than 7 consecutive business days, the agent may initiate "
            "a carrier trace. If transit exceeds 14 days past the estimated delivery window, a full replacement or refund may be authorized."
        ),
        "max_auto_refund_limit": 100.00,
        "return_window_days": 14
    },
    {
        "policy_id": "POL-WAR-04",
        "category": "warranty",
        "title": "Electronics 1-Year Limited Warranty",
        "content": (
            "All hardware products (cameras, audio equipment, electronics) carry a 1-year limited manufacturer warranty. "
            "Issues arising after the 30-day initial return window qualify for free manufacturer repair or replacement rather than direct refund."
        ),
        "max_auto_refund_limit": 50.00,
        "return_window_days": 365
    },
    {
        "policy_id": "POL-ACC-05",
        "category": "cancellation",
        "title": "Order Cancellation and Address Changes",
        "content": (
            "Orders in PROCESSING status may be canceled or have their shipping address updated immediately. Once an order reaches "
            "SHIPPED status, the delivery address cannot be changed and cancellation cannot be processed; the customer must await delivery "
            "and initiate a return."
        ),
        "max_auto_refund_limit": 500.00,
        "return_window_days": 0
    }
]

PRODUCT_CATALOG = [
    {"sku": "SKU-CAM-4K", "title": "UltraHD Action Cam", "price": 120.00, "category": "electronics"},
    {"sku": "SKU-CASE-1", "title": "Shockproof Camera Case", "price": 12.00, "category": "accessories"},
    {"sku": "SKU-HEAD-NC", "title": "Wireless Noise-Canceling Headphones", "price": 149.99, "category": "audio"},
    {"sku": "SKU-USB-C", "title": "Braided USB-C Fast Charger Cable (2m)", "price": 9.99, "category": "cables"},
    {"sku": "SKU-MNT-PRO", "title": "Ergonomic Aluminum Laptop Stand", "price": 45.50, "category": "office"},
    {"sku": "SKU-KB-RGB", "title": "Compact Mechanical Keyboard", "price": 89.00, "category": "electronics"},
    {"sku": "SKU-MOU-WL", "title": "Precision Optical Wireless Mouse", "price": 34.00, "category": "electronics"},
    {"sku": "SKU-SPK-BT", "title": "Waterproof Portable Bluetooth Speaker", "price": 59.95, "category": "audio"},
    {"sku": "SKU-DRN-MINI", "title": "Foldable Quadcopter Mini Drone", "price": 249.00, "category": "electronics"},
    {"sku": "SKU-BAG-TECH", "title": "Water-Resistant Commuter Backpack", "price": 75.00, "category": "apparel"},
    {"sku": "SKU-PWR-20K", "title": "20000mAh PD Fast Power Bank", "price": 39.99, "category": "electronics"},
    {"sku": "SKU-LED-DSK", "title": "Dimmable Smart Desk Light Bar", "price": 28.50, "category": "office"},
    {"sku": "SKU-STBL-3AX", "title": "3-Axis Handheld Smartphone Gimbal", "price": 115.00, "category": "electronics"},
    {"sku": "SKU-MAT-DESK", "title": "Vegan Leather Large Desk Pad", "price": 18.00, "category": "office"},
    {"sku": "SKU-SCRN-CL", "title": "Antimicrobial Screen Cleaning Kit", "price": 7.50, "category": "accessories"}
]

FIRST_NAMES = ["Liam", "Olivia", "Noah", "Emma", "Oliver", "Charlotte", "Elijah", "Amelia", "James", "Sophia",
               "William", "Isabella", "Benjamin", "Mia", "Lucas", "Evelyn", "Henry", "Harper", "Alexander", "Camila"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

TIERS = ["Standard", "Silver", "Gold", "VIP_Platinum"]
TIER_WEIGHTS = [0.55, 0.25, 0.15, 0.05]


async def bootstrap_database_schema(pool: AsyncConnectionPool) -> None:
    """
    Idempotently creates all required tables and full-text search indexes on application startup.
    Ensures the service runs seamlessly on brand new online cloud databases (Neon, Supabase, Render).
    """
    logger.info("Verifying and bootstrapping database schema...")
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # 1. Customers Table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    tier VARCHAR(20) NOT NULL DEFAULT 'Standard',
                    trust_score NUMERIC(3, 2) NOT NULL DEFAULT 0.95,
                    total_orders_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # 2. Orders Table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id VARCHAR(50) PRIMARY KEY,
                    customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
                    status VARCHAR(30) NOT NULL,
                    order_total NUMERIC(10, 2) NOT NULL,
                    items JSONB NOT NULL,
                    tracking_number VARCHAR(100),
                    order_date TIMESTAMPTZ NOT NULL,
                    delivered_date TIMESTAMPTZ,
                    is_return_eligible BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # 3. Policies Table with Full-Text Search
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS policies (
                    policy_id VARCHAR(50) PRIMARY KEY,
                    category VARCHAR(50) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    content TEXT NOT NULL,
                    max_auto_refund_limit NUMERIC(10, 2) NOT NULL DEFAULT 100.00,
                    return_window_days INT NOT NULL DEFAULT 30,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # 4. Refund Transactions Audit Table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS refund_transactions (
                    transaction_id VARCHAR(100) PRIMARY KEY,
                    order_id VARCHAR(50) NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
                    amount NUMERIC(10, 2) NOT NULL,
                    reason TEXT,
                    idempotency_key VARCHAR(100) UNIQUE,
                    customer_id VARCHAR(50),
                    status VARCHAR(30) NOT NULL DEFAULT 'SETTLED',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

            # 5. Performance & Full-Text Search Indexes
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                CREATE INDEX IF NOT EXISTS idx_refund_order ON refund_transactions(order_id);
                CREATE INDEX IF NOT EXISTS idx_policies_fts ON policies USING gin(to_tsvector('english', title || ' ' || content));
            """)

            # 6. Check if Policies exist
            await cur.execute("SELECT COUNT(*) FROM policies;")
            pol_count = (await cur.fetchone())[0]
            if pol_count == 0:
                logger.info("Policies table is empty. Seeding official business policies...")
                for p in INITIAL_POLICIES:
                    await cur.execute("""
                        INSERT INTO policies (policy_id, category, title, content, max_auto_refund_limit, return_window_days)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (policy_id) DO NOTHING;
                    """, (
                        p["policy_id"], p["category"], p["title"], p["content"],
                        p["max_auto_refund_limit"], p["return_window_days"]
                    ))

            # 7. Check if Customers exist
            await cur.execute("SELECT COUNT(*) FROM customers;")
            cust_count = (await cur.fetchone())[0]
            if cust_count == 0:
                logger.info("Customers table is empty. Seeding 50 customer profiles and 150 orders...")
                await _seed_initial_data(cur)

    logger.info("Database schema and seed verification complete.")


async def _seed_initial_data(cur) -> None:
    """Populates realistic customer and order data for simulation and interview demos."""
    random.seed(42)
    now = datetime.now(timezone.utc)

    # 1. Ensure deterministic test customer CUST-1001 exists with standard test order ORD-9821
    await cur.execute("""
        INSERT INTO customers (customer_id, name, email, tier, trust_score, total_orders_count, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id) DO NOTHING;
    """, (
        "CUST-1001",
        "Alexander Wright",
        "alexander.wright@example.com",
        "Gold",
        0.98,
        4,
        now - timedelta(days=120)
    ))

    # Add specific realistic test orders for CUST-1001
    await cur.execute("""
        INSERT INTO orders (order_id, customer_id, status, order_total, items, tracking_number, order_date, delivered_date, is_return_eligible)
        VALUES 
        ('ORD-9821', 'CUST-1001', 'DELIVERED', 120.00, %s, 'TRK-9821-DEL', %s, %s, TRUE),
        ('ORD-1002', 'CUST-1001', 'DELIVERED', 45.50, %s, 'TRK-1002-DEL', %s, %s, TRUE),
        ('ORD-1003', 'CUST-1001', 'SHIPPED', 89.00, %s, 'TRK-1003-SHP', %s, NULL, FALSE),
        ('ORD-1004', 'CUST-1001', 'DELIVERED', 149.99, %s, 'TRK-1004-OLD', %s, %s, FALSE)
        ON CONFLICT (order_id) DO NOTHING;
    """, (
        json.dumps([{"sku": "SKU-CAM-4K", "title": "UltraHD Action Cam", "price": 120.00, "qty": 1}]),
        now - timedelta(days=12), now - timedelta(days=8),
        json.dumps([{"sku": "SKU-MNT-PRO", "title": "Ergonomic Aluminum Laptop Stand", "price": 45.50, "qty": 1}]),
        now - timedelta(days=20), now - timedelta(days=16),
        json.dumps([{"sku": "SKU-KB-RGB", "title": "Compact Mechanical Keyboard", "price": 89.00, "qty": 1}]),
        now - timedelta(days=2),
        json.dumps([{"sku": "SKU-HEAD-NC", "title": "Wireless Noise-Canceling Headphones", "price": 149.99, "qty": 1}]),
        now - timedelta(days=75), now - timedelta(days=70)
    ))

    # 2. Seed remaining 49 customers and ~146 orders
    customer_ids = ["CUST-1001"]
    for i in range(1002, 1051):
        c_id = f"CUST-{i}"
        customer_ids.append(c_id)
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}.{i}@example.com"
        tier = random.choices(TIERS, weights=TIER_WEIGHTS)[0]
        trust = round(random.uniform(0.70, 0.99), 2)
        created_days_ago = random.randint(30, 365)
        created_at = now - timedelta(days=created_days_ago)

        await cur.execute("""
            INSERT INTO customers (customer_id, name, email, tier, trust_score, total_orders_count, created_at)
            VALUES (%s, %s, %s, %s, %s, 0, %s)
            ON CONFLICT (customer_id) DO NOTHING;
        """, (c_id, name, email, tier, trust, created_at))

    # Seed remaining orders
    order_seq = 2000
    for c_id in customer_ids[1:]:
        num_orders = random.choices([1, 2, 3, 4, 5], weights=[0.30, 0.35, 0.20, 0.10, 0.05])[0]
        for _ in range(num_orders):
            order_seq += 1
            ord_id = f"ORD-{order_seq}"
            num_items = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            chosen_prods = random.sample(PRODUCT_CATALOG, k=num_items)
            items_payload = []
            total = 0.0
            for prod in chosen_prods:
                qty = random.choices([1, 2], weights=[0.85, 0.15])[0]
                line_total = prod["price"] * qty
                total += line_total
                items_payload.append({
                    "sku": prod["sku"],
                    "title": prod["title"],
                    "price": prod["price"],
                    "quantity": qty,
                    "line_total": round(line_total, 2)
                })
            total = round(total, 2)

            days_ago = random.randint(1, 90)
            order_date = now - timedelta(days=days_ago)
            status_choice = random.choices(["DELIVERED", "SHIPPED", "PROCESSING"], weights=[0.75, 0.15, 0.10])[0]

            delivered_date = None
            is_return_eligible = False
            if status_choice == "DELIVERED":
                delivered_days_ago = max(0, days_ago - random.randint(2, 5))
                delivered_date = now - timedelta(days=delivered_days_ago)
                is_return_eligible = (delivered_days_ago <= 30)

            tracking = f"TRK-{order_seq}-{status_choice[:3]}"

            await cur.execute("""
                INSERT INTO orders (order_id, customer_id, status, order_total, items, tracking_number, order_date, delivered_date, is_return_eligible)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO NOTHING;
            """, (
                ord_id, c_id, status_choice, total, json.dumps(items_payload),
                tracking, order_date, delivered_date, is_return_eligible
            ))

    # Update customer total_orders_count
    await cur.execute("""
        UPDATE customers c
        SET total_orders_count = sub.cnt
        FROM (
            SELECT customer_id, count(*) as cnt
            FROM orders
            GROUP BY customer_id
        ) sub
        WHERE c.customer_id = sub.customer_id;
    """)
