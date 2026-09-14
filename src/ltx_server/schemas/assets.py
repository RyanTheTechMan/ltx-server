from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class AssetInfo(BaseModel):
    id: str
    type: Literal["image", "audio", "video"]
    mime_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    has_audio: bool = False
