import os
import sys
import asyncio
import selectors
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

def selector_loop_factory(*args, **kwargs):
    return asyncio.SelectorEventLoop(selectors.SelectSelector())

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        import uvicorn.loops.asyncio
        import uvicorn.loops.auto
        uvicorn.loops.asyncio.asyncio_loop_factory = lambda *a, **kw: selector_loop_factory
        uvicorn.loops.auto.auto_loop_factory = lambda *a, **kw: selector_loop_factory
    except ImportError:
        pass

from core.config import get_settings
from core.db_bootstrap import bootstrap_database_schema
from core.db import get_db_pool
from core.redis_client import redis_client
from core.graph import build_resolveops_graph
from api.routes import router as webhook_router
from api.chat_routes import router as chat_router
from api.dashboard_routes import router as dashboard_router

logger = logging.getLogger("resolveops.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Context Manager.
    1. Initializes resilient PostgreSQL connection pool.
    2. Idempotently provisions tables, indexes, policies, and initial dataset.
    3. Initializes LangGraph Checkpointer (short-term) and Store (long-term).
    4. Compiles and mounts the autonomous resolution StateGraph.
    """
    logger.info("Initializing ResolveOps Engine...")
    async with AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        max_size=20,
        kwargs={"autocommit": True}
    ) as pool:
        # Idempotent database bootstrap (tables, full-text indexes, catalog seed)
        await bootstrap_database_schema(pool)

        # Checkpointer for short-term thread state retention
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        # Cross-thread Store for durable customer profile and dispute facts
        store = AsyncPostgresStore(pool)
        await store.setup()

        # Mount compiled graph and store in application state
        app.state.resolveops_graph = build_resolveops_graph(checkpointer=checkpointer, store=store)
        app.state.resolveops_store = store

        logger.info("ResolveOps Engine ready to accept traffic.")
        yield


app = FastAPI(
    title="ResolveOps Agent Engine",
    description="Autonomous customer resolution engine with LangGraph, PostgreSQL full-text search, and Redis mutex locking.",
    version="2.0.0",
    lifespan=lifespan
)

# Mount application routers
app.include_router(chat_router)
app.include_router(webhook_router)
app.include_router(dashboard_router)


@app.get("/healthz", tags=["Observability & Probes"])
@app.get("/health", tags=["Observability & Probes"])
async def liveness_probe():
    """Liveness probe for container orchestrators (Kubernetes / Render / Railway)."""
    return {"status": "LIVE", "service": "ResolveOps Engine", "environment": settings.APP_ENV}


@app.get("/readyz", tags=["Observability & Probes"])
async def readiness_probe():
    """Readiness probe checking live PostgreSQL and Redis connectivity."""
    checks = {"postgres": "UNKNOWN", "redis": "UNKNOWN"}

    # PostgreSQL Check
    try:
        pool = get_db_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1;")
        checks["postgres"] = "CONNECTED"
    except Exception as exc:
        checks["postgres"] = f"ERROR: {str(exc)}"

    # Redis Check
    try:
        await redis_client.ping()
        checks["redis"] = "CONNECTED"
    except Exception as exc:
        checks["redis"] = f"ERROR: {str(exc)}"

    is_ready = (checks["postgres"] == "CONNECTED" and checks["redis"] == "CONNECTED")
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "READY" if is_ready else "DEGRADED",
            "checks": checks
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=(settings.APP_ENV == "development"),
        loop="main:selector_loop_factory"
    )
