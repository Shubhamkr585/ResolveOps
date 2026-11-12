import asyncio
import redis.asyncio as aioredis
from typing import Optional, Dict
from core.config import get_settings

settings = get_settings()

# Safe lock release script: deletes key ONLY if value matches the owner token
RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

_clients: Dict[int, aioredis.Redis] = {}


def get_redis_client() -> aioredis.Redis:
    """
    Returns an async Redis client bound to the current running event loop.
    Prevents 'Event loop is closed' errors during multi-test execution or process restarts.
    Supports redis:// and rediss:// (TLS) URLs.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    loop_key = id(loop) if loop is not None else 0
    if loop_key not in _clients:
        _clients[loop_key] = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _clients[loop_key]


class RedisClientProxy:
    """Transparent proxy delegating calls to the loop-scoped Redis client."""

    def __getattr__(self, name: str):
        client = get_redis_client()
        return getattr(client, name)


redis_client = RedisClientProxy()
