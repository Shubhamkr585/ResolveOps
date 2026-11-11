import hmac
import hashlib
import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException, Request, status

load_dotenv()

def get_webhook_secret() -> bytes:
    return os.getenv("ADMIN_WEBHOOK_SECRET", "e077b39f838ff7d99e8725bb4c1999abf4d4bfd8902e922991401b66bc91045a").encode()


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