import os
import sys
import asyncio
import selectors

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

from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from api.routes import router as webhook_router
from api.chat_routes import router as chat_router
from core.graph import build_resolveops_graph

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/resolveops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # PostgreSQL connection pool for LangGraph checkpoints (short-term) and Store (long-term)
    async with AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()  # Auto-creates 'checkpoints' & 'checkpoint_blobs' tables

        store = AsyncPostgresStore(pool)
        await store.setup()  # Auto-creates 'store' tables for cross-thread long-term memory

        app.state.resolveops_graph = build_resolveops_graph(checkpointer=checkpointer, store=store)
        app.state.resolveops_store = store
        yield

app = FastAPI(title="ResolveOps Agent Engine", lifespan=lifespan)

# Register endpoints
app.include_router(chat_router)
app.include_router(webhook_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ResolveOps Engine"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, loop="main:selector_loop_factory")