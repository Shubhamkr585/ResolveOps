import os
import json
import random
from datetime import datetime, timedelta, timezone
import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5433/resolveops")

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

POLICIES = [
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


def create_schema(conn: psycopg.Connection):
    with conn.cursor() as cur:
        # Customers Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL,
                tier VARCHAR(30) NOT NULL,
                trust_score FLOAT NOT NULL,
                total_orders_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Orders Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id VARCHAR(50) PRIMARY KEY,
                customer_id VARCHAR(50) NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
                status VARCHAR(50) NOT NULL,
                order_total NUMERIC(10, 2) NOT NULL,
                items JSONB NOT NULL,
                order_date TIMESTAMPTZ NOT NULL,
                delivered_date TIMESTAMPTZ,
                payment_method VARCHAR(50) NOT NULL,
                tracking_number VARCHAR(100),
                is_return_eligible BOOLEAN NOT NULL DEFAULT TRUE
            );
        """)

        # Policies Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS policies (
                policy_id VARCHAR(50) PRIMARY KEY,
                category VARCHAR(50) NOT NULL,
                title VARCHAR(200) NOT NULL,
                content TEXT NOT NULL,
                max_auto_refund_limit NUMERIC(10, 2) NOT NULL,
                return_window_days INT NOT NULL
            );
        """)

        # Refund Transactions Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS refund_transactions (
                transaction_id VARCHAR(100) PRIMARY KEY,
                order_id VARCHAR(50) NOT NULL,
                customer_id VARCHAR(50) NOT NULL,
                amount NUMERIC(10, 2) NOT NULL,
                status VARCHAR(50) NOT NULL,
                reason TEXT,
                idempotency_key VARCHAR(100) UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        conn.commit()
    print("Database schema verified and created.")


def seed_data(conn: psycopg.Connection):
    random.seed(42)  # Deterministic seed for reproducible testing
    now = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        # 1. Seed Policies
        for p in POLICIES:
            cur.execute("""
                INSERT INTO policies (policy_id, category, title, content, max_auto_refund_limit, return_window_days)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (policy_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    max_auto_refund_limit = EXCLUDED.max_auto_refund_limit,
                    return_window_days = EXCLUDED.return_window_days;
            """, (p["policy_id"], p["category"], p["title"], p["content"], p["max_auto_refund_limit"], p["return_window_days"]))
        print(f"Seeded {len(POLICIES)} policy documents.")

        # 2. Seed 50 Customers (CUST-1001 to CUST-1050)
        customer_ids = []
        for i in range(1, 51):
            c_id = f"CUST-{1000 + i}"
            customer_ids.append(c_id)
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            name = f"{first} {last}"
            email = f"{first.lower()}.{last.lower()}{i}@example.com"
            tier = random.choices(TIERS, weights=TIER_WEIGHTS, k=1)[0]
            trust = round(random.uniform(0.70, 0.99), 2)

            cur.execute("""
                INSERT INTO customers (customer_id, name, email, tier, trust_score, total_orders_count, created_at)
                VALUES (%s, %s, %s, %s, %s, 0, %s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    tier = EXCLUDED.tier,
                    trust_score = EXCLUDED.trust_score;
            """, (c_id, name, email, tier, trust, now - timedelta(days=random.randint(60, 400))))

        print("Seeded 50 customer profiles.")

        # 3. Seed 150 Orders (ORD-1001 to ORD-1150) + Benchmark ORD-9821
        order_statuses = ["DELIVERED", "DELIVERED", "DELIVERED", "SHIPPED", "PROCESSING", "CANCELLED"]

        # Ensure benchmark order ORD-9821 exists specifically for test consistency
        benchmark_order_items = [
            {"sku": "SKU-CAM-4K", "title": "UltraHD Action Cam", "price": 120.00, "category": "electronics"},
            {"sku": "SKU-CASE-1", "title": "Shockproof Camera Case", "price": 12.00, "category": "accessories"}
        ]
        cur.execute("""
            INSERT INTO orders (order_id, customer_id, status, order_total, items, order_date, delivered_date, payment_method, tracking_number, is_return_eligible)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (order_id) DO UPDATE SET
                status = EXCLUDED.status,
                order_total = EXCLUDED.order_total,
                items = EXCLUDED.items,
                is_return_eligible = EXCLUDED.is_return_eligible;
        """, (
            "ORD-9821",
            "CUST-1001",
            "DELIVERED",
            132.00,
            json.dumps(benchmark_order_items),
            now - timedelta(days=5),
            now - timedelta(days=3),
            "CREDIT_CARD",
            "TRK-9821-USPS",
            True
        ))

        # Generate remaining 149 diverse orders
        for i in range(1, 150):
            ord_id = f"ORD-{1000 + i}"
            cust_id = random.choice(customer_ids)
            status = random.choice(order_statuses)

            # Pick 1 to 3 items
            item_count = random.randint(1, 3)
            selected_items = random.sample(PRODUCT_CATALOG, item_count)
            items_payload = [
                {"sku": it["sku"], "title": it["title"], "price": it["price"], "category": it["category"], "quantity": 1}
                for it in selected_items
            ]
            total_price = sum(it["price"] for it in items_payload)

            order_age_days = random.randint(2, 60)
            order_date = now - timedelta(days=order_age_days)
            delivered_date = order_date + timedelta(days=random.randint(2, 5)) if status == "DELIVERED" else None

            # Eligible for return only if delivered within 30 days
            is_return_eligible = False
            if status == "DELIVERED" and delivered_date:
                days_since_delivery = (now - delivered_date).days
                is_return_eligible = days_since_delivery <= 30

            cur.execute("""
                INSERT INTO orders (order_id, customer_id, status, order_total, items, order_date, delivered_date, payment_method, tracking_number, is_return_eligible)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (order_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    order_total = EXCLUDED.order_total,
                    items = EXCLUDED.items,
                    is_return_eligible = EXCLUDED.is_return_eligible;
            """, (
                ord_id,
                cust_id,
                status,
                total_price,
                json.dumps(items_payload),
                order_date,
                delivered_date,
                random.choice(["CREDIT_CARD", "PAYPAL", "APPLE_PAY"]),
                f"TRK-{ord_id}-FEDEX" if status in ("DELIVERED", "SHIPPED") else None,
                is_return_eligible
            ))

        # Update customer total order counts
        cur.execute("""
            UPDATE customers c
            SET total_orders_count = (
                SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id
            );
        """)
        conn.commit()
    print("Seeded 150 orders across customer database successfully.")


def main():
    print(f"Connecting to database: {DB_URI}")
    with psycopg.connect(DB_URI) as conn:
        create_schema(conn)
        seed_data(conn)
    print("Database seeding completed.")


if __name__ == "__main__":
    main()
