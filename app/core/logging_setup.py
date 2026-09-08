"""Logging: extraction.log, errors.log, unsupported.log + console."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FMT = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"

_configured = False


class _LevelFilter(logging.Filter):
    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


def setup_logging(log_dir="logs", level: str = "INFO", console: bool = True) -> logging.Logger:
    """Idempotent. Returns the package root logger."""
    global _configured
    root = logging.getLogger("app")
    root.setLevel(logging.DEBUG)
    if _configured:
        root.setLevel(logging.DEBUG)
        for h in root.handlers:
            if getattr(h, "_is_console", False):
                h.setLevel(getattr(logging, level.upper(), logging.INFO))
        return root

    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(FMT, DATEFMT)

    main = RotatingFileHandler(d / "extraction.log", maxBytes=8 << 20, backupCount=3, encoding="utf-8")
    main.setLevel(logging.DEBUG)
    main.setFormatter(fmt)
    root.addHandler(main)

    errors = RotatingFileHandler(d / "errors.log", maxBytes=4 << 20, backupCount=3, encoding="utf-8")
    errors.setLevel(logging.WARNING)
    errors.setFormatter(fmt)
    root.addHandler(errors)

    unsupported = RotatingFileHandler(d / "unsupported.log", maxBytes=4 << 20, backupCount=2, encoding="utf-8")
    unsupported.setFormatter(fmt)
    unsupported.addFilter(logging.Filter("app.unsupported"))
    logging.getLogger("app.unsupported").addHandler(unsupported)

    if console:
        con = logging.StreamHandler()
        con.setLevel(getattr(logging, level.upper(), logging.INFO))
        con.setFormatter(fmt)
        con._is_console = True  # type: ignore[attr-defined]
        root.addHandler(con)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger("app." + name if not name.startswith("app") else name)


def unsupported_logger() -> logging.Logger:
    return logging.getLogger("app.unsupported")
