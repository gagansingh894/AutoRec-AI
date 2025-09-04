import sys
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),        # timestamps
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
)

logger = structlog.get_logger()