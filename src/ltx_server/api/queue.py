from typing import Annotated

from fastapi import APIRouter, Depends

from ltx_server.api.dependencies import jobs
from ltx_server.jobs.manager import JobManager, QueueInfo

router = APIRouter()


@router.get("/queue", response_model=QueueInfo, tags=["diagnostics"])
async def queue(manager: Annotated[JobManager, Depends(jobs)]) -> QueueInfo:
    """Waiting jobs in FIFO order; length and max_size exclude the running job."""
    return manager.queue_info()
