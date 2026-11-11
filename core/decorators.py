import asyncio
import functools
import logging
import uuid
from enum import Enum
from typing import Callable, Any, Optional
from pydantic import BaseModel

from core.redis_client import redis_client, RELEASE_LOCK_LUA

logger = logging.getLogger("resolveops.decorators")


class IdempotencyStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CachedActionResponse(BaseModel):
    action_id: str
    status: IdempotencyStatus
    response_data: Optional[Any] = None
    error_message: Optional[str] = None


class LockAcquisitionError(Exception):
    pass


class ActionInProgressError(Exception):
    pass


def transactional_tool(
    entity_key_param: str,
    action_id_param: str = "action_id",
    lock_ttl_seconds: int = 10,
    lock_retry_times: int = 3,
    lock_retry_delay_ms: int = 150,
    idempotency_ttl_seconds: int = 86400,
):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            entity_id = kwargs.get(entity_key_param)
            action_id = kwargs.get(action_id_param)

            if not entity_id or not action_id:
                raise ValueError(
                    f"Tool {func.__name__} requires '{entity_key_param}' and '{action_id_param}'."
                )

            idempotency_redis_key = f"idempotency:{action_id}"
            lock_redis_key = f"lock:entity:{entity_id}"
            owner_token = str(uuid.uuid4())

            # 1. Fast Check Idempotency Store
            raw_cached = await redis_client.get(idempotency_redis_key)
            if raw_cached:
                cached_action = CachedActionResponse.model_validate_json(raw_cached)
                if cached_action.status == IdempotencyStatus.COMPLETED:
                    logger.info(f"[IDEMPOTENCY_HIT] Returning cached result for {action_id}")
                    return cached_action.response_data
                if cached_action.status == IdempotencyStatus.IN_PROGRESS:
                    raise ActionInProgressError(f"Action {action_id} is actively executing.")

            # 2. Acquire Distributed Mutex
            acquired = False
            for _ in range(lock_retry_times):
                acquired = await redis_client.set(
                    lock_redis_key,
                    owner_token,
                    nx=True,
                    ex=lock_ttl_seconds
                )
                if acquired:
                    break
                await asyncio.sleep(lock_retry_delay_ms / 1000.0)

            if not acquired:
                raise LockAcquisitionError(f"Resource {entity_id} is currently locked.")

            try:
                # 3. Mark in-progress
                in_progress = CachedActionResponse(
                    action_id=action_id,
                    status=IdempotencyStatus.IN_PROGRESS
                )
                await redis_client.set(
                    idempotency_redis_key,
                    in_progress.model_dump_json(),
                    ex=lock_ttl_seconds * 2
                )

                # 4. Execute Side-Effect
                result = await func(*args, **kwargs)

                # 5. Commit Completed Status
                completed = CachedActionResponse(
                    action_id=action_id,
                    status=IdempotencyStatus.COMPLETED,
                    response_data=result
                )
                await redis_client.set(
                    idempotency_redis_key,
                    completed.model_dump_json(),
                    ex=idempotency_ttl_seconds
                )
                return result

            except Exception as exc:
                failed = CachedActionResponse(
                    action_id=action_id,
                    status=IdempotencyStatus.FAILED,
                    error_message=str(exc)
                )
                await redis_client.set(
                    idempotency_redis_key,
                    failed.model_dump_json(),
                    ex=300
                )
                raise exc

            finally:
                # 6. Release lock atomically
                await redis_client.eval(RELEASE_LOCK_LUA, 1, lock_redis_key, owner_token)

        return wrapper
    return decorator