from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event
from typing import Protocol

from ltx_server.jobs.state import JobInfo
from ltx_server.schemas.generation import GenerationSpec


@dataclass
class Job:
    info: JobInfo
    spec: GenerationSpec
    cancel: Event = field(default_factory=Event)
    readers: int = 0
    downloaded_at: datetime | None = None
    delete_pending: bool = False


class JobRepository(Protocol):
    def add(self, job: Job) -> None: ...
    def get(self, identifier: str) -> Job | None: ...
    def all(self) -> Iterable[Job]: ...
    def remove(self, identifier: str) -> None: ...


class MemoryJobRepository:
    """Event-loop-owned metadata; intentionally not persistent in phase 1."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job) -> None:
        if job.info.id in self._jobs:
            raise ValueError("Duplicate job id")
        self._jobs[job.info.id] = job

    def get(self, identifier: str) -> Job | None:
        return self._jobs.get(identifier)

    def all(self) -> Iterable[Job]:
        return tuple(self._jobs.values())

    def remove(self, identifier: str) -> None:
        self._jobs.pop(identifier, None)
