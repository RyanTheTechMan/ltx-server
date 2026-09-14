from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

BODY_TOO_LARGE = "Upload exceeds request body limit"


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].rstrip("/") != "/v1/assets":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        try:
            length = int(headers.get("content-length", "0"))
            too_large = length < 0 or length > self.limit
        except ValueError:
            too_large = True
        if too_large:
            await JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "UPLOAD_TOO_LARGE",
                        "message": BODY_TOO_LARGE,
                    }
                },
            )(scope, receive, send)
            return
        consumed = 0

        async def bounded_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.limit:
                    # The multipart parser closes all temporary files for this exception.
                    raise MultiPartException(BODY_TOO_LARGE)
            return message

        await self.app(scope, bounded_receive, send)
