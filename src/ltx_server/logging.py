import json
import logging
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        values: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "metrics",
            "job_id",
            "status",
            "error_code",
            "width",
            "height",
            "frames",
            "fps",
            "seed",
            "total_seconds",
        ):
            if hasattr(record, key):
                values[key] = getattr(record, key)
        if record.exc_info:
            values["exception"] = self.formatException(record.exc_info)
        return json.dumps(values)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger = logging.getLogger("ltx_server")
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
