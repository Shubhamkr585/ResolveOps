import hmac
import hashlib
import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException, Request, status

load_dotenv()

def get_webhook_secret() -> bytes:
    secret = os.getenv("ADMIN_WEBHOOK_SECRET")
    if not secret:
        # Default development secret; override with ADMIN_WEBHOOK_SECRET in production
        secret = "resolveops_default_development_secret_change_in_production"
    return secret.encode()


async def verify_admin_signature(
    request: Request,
    x_resolveops_signature: str = Header(..., alias="X-ResolveOps-Signature")
):
    body = await request.body()
    secret = get_webhook_secret()
    computed = hmac.new(secret, body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(f"sha256={computed}", x_resolveops_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid HMAC webhook signature."
        )