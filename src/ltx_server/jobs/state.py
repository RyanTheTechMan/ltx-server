from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from ltx_server.errors import ErrorDetail


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    LOADING = "loading"
    ENCODING = "encoding"
    GENERATING = "generating"
    DECODING = "decoding"
    ENCODING_OUTPUT = "encoding_output"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


STAGES = {
    JobStatus.QUEUED: 0,
    JobStatus.LOADING: 5,
    JobStatus.ENCODING: 10,
    JobStatus.GENERATING: 15,
    JobStatus.DECODING: 90,
    JobStatus.ENCODING_OUTPUT: 95,
    JobStatus.COMPLETE: 100,
}
TERMINAL = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED}


class OutputInfo(BaseModel):
    content_url: str = ""
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    duration: float = Field(gt=0, allow_inf_nan=False)
    has_audio: bool
    size_bytes: int = Field(default=0, ge=0)


class JobInfo(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
    cancellation_requested: bool = False
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    output: OutputInfo | None = None
    error: ErrorDetail | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    def transition(self, status: JobStatus, now: datetime | None = None) -> None:
        if status == self.status:
            return
        if self.status in TERMINAL:
            if self.status != JobStatus.COMPLETE or status not in {
                JobStatus.EXPIRED,
                JobStatus.CANCELLED,
            }:
                raise ValueError(f"Invalid transition: {self.status} -> {status}")
        elif status == JobStatus.EXPIRED or (
            status in STAGES and STAGES[status] <= STAGES[self.status]
        ):
            raise ValueError(f"Invalid transition: {self.status} -> {status}")
        self.status = status
        self.progress = STAGES.get(status, self.progress)
        now = now or utcnow()
        if status not in TERMINAL and status != JobStatus.QUEUED and self.started_at is None:
            self.started_at = now
        if status in TERMINAL:
            self.completed_at = now
