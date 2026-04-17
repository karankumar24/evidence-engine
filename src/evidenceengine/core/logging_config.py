"""Configure structured JSON logging for production use."""

import logging
import sys


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    try:
        from pythonjsonlogger import jsonlogger

        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    except ImportError:
        formatter = logging.Formatter(  # type: ignore[assignment]
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)

    # Quiet noisy libraries in non-debug mode
    if not debug:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
