import hmac
import hashlib
from fastapi import Header, HTTPException, Request, status
from core.config import get_settings

settings = get_settings()


def get_webhook_secret() -> bytes:
    """Retrieves secret key for HMAC-SHA256 signature verification."""
    return settings.ADMIN_WEBHOOK_SECRET.encode()


async def verify_admin_signature(
    request: Request,
    x_resolveops_signature: str = Header(..., alias="X-ResolveOps-Signature")
):
    body = await request.body()
    secret = get_webhook_secret()
    computed = hmac.new(secret, body, hashlib.sha256).hexdigest()

    # Supports either raw hex or sha256=<hex>
    expected_prefixed = f"sha256={computed}"
    if not (hmac.compare_digest(expected_prefixed, x_resolveops_signature) or
            hmac.compare_digest(computed, x_resolveops_signature)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid HMAC webhook signature."
        )
