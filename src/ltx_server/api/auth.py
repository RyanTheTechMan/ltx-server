import hmac
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ltx_server.errors import ErrorCode, ServiceError

bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    settings = request.app.state.settings
    if request.url.path == "/v1/health" and not settings.health_requires_auth:
        return
    secret = settings.api_key.get_secret_value()
    if not secret:
        return
    candidate = credentials.credentials if credentials else ""
    if not hmac.compare_digest(candidate.encode("utf-8"), secret.encode("utf-8")):
        raise ServiceError(ErrorCode.UNAUTHORIZED, "Valid Bearer token required", 401)
