import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from starlette.datastructures import UploadFile

from ltx_server.config import Settings
from ltx_server.errors import ErrorCode, ServiceError
from ltx_server.jobs.state import utcnow
from ltx_server.media.storage import Storage, new_id
from ltx_server.media.validation import validate_media
from ltx_server.schemas.assets import AssetInfo


@dataclass
class AssetRecord:
    info: AssetInfo
    leases: set[str] = field(default_factory=set)


class AssetManager:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.records: dict[str, AssetRecord] = {}
        self.uploading: set[str] = set()

    async def upload(self, file: UploadFile) -> AssetInfo:
        identifier = new_id("asset")
        self.uploading.add(identifier)
        try:
            path = self.storage.create_partial(identifier)
            size = 0
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.settings.upload_body_limit - 64 * 1024:
                        raise ServiceError(
                            ErrorCode.UPLOAD_TOO_LARGE, "Upload exceeds size limit", 413
                        )
                    await asyncio.to_thread(target.write, chunk)
            if not size:
                raise ServiceError(ErrorCode.INVALID_MEDIA, "Empty upload", 415)
            media = await validate_media(path, self.settings)
            limit = getattr(self.settings, f"max_{media.type}_upload_mb") * 1024 * 1024
            if size > limit:
                raise ServiceError(
                    ErrorCode.UPLOAD_TOO_LARGE, f"{media.type} exceeds size limit", 413
                )
            now = utcnow()
            info = AssetInfo(
                id=identifier,
                type=media.type,
                mime_type=media.mime_type,
                size_bytes=size,
                created_at=now,
                expires_at=now + timedelta(seconds=self.settings.asset_ttl_seconds),
                width=media.width,
                height=media.height,
                duration=media.duration,
            )
            self.storage.publish(identifier, "assets")
            self.records[identifier] = AssetRecord(info)
            return info
        finally:
            self.uploading.discard(identifier)
            self.storage.remove("tmp", f"{identifier}.partial")
            await file.close()

    def get(self, identifier: str, now: datetime | None = None) -> AssetRecord:
        record = self.records.get(identifier)
        if record is None:
            raise ServiceError(ErrorCode.ASSET_NOT_FOUND, "Asset not found", 404)
        if record.info.expires_at <= (now or utcnow()):
            raise ServiceError(ErrorCode.ASSET_EXPIRED, "Asset has expired", 410)
        if not self.storage.path("assets", f"{identifier}.bin").is_file():
            raise ServiceError(ErrorCode.ASSET_NOT_FOUND, "Asset file not found", 404)
        return record

    def acquire(self, job_id: str, expected: dict[str, str]) -> None:
        # Validate everything before mutating leases. No awaits: atomic on the event loop.
        records = [self.get(identifier) for identifier in expected]
        if any(record.info.type != expected[record.info.id] for record in records):
            raise ServiceError(ErrorCode.INVALID_INPUT, "An asset has the wrong media type", 422)
        for record in records:
            record.leases.add(job_id)

    def release(self, job_id: str) -> None:
        for record in self.records.values():
            record.leases.discard(job_id)

    def delete(self, identifier: str) -> None:
        record = self.records.get(identifier)
        if record is None:
            return
        if record.leases:
            raise ServiceError(
                ErrorCode.ASSET_IN_USE, "Asset is held by a queued or running job", 409
            )
        self.storage.remove("assets", f"{identifier}.bin")
        del self.records[identifier]
