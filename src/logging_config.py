import logging
import sys

try:
    from loguru import logger as _loguru_logger
except ModuleNotFoundError:  # pragma: no cover - fallback path
    _loguru_logger = None


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
PLAIN_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)


if _loguru_logger is not None:

    logger = _loguru_logger

    class InterceptHandler(logging.Handler):
        """Route standard logging records into loguru so libraries stay consistent."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except Exception:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    def setup_logging(
        level: str = "INFO",
        sink=None,
        log_format: str = LOG_FORMAT,
        enqueue: bool = True,
    ) -> None:
        """Initialize loguru with a configurable sink and intercept stdlib logging."""
        logger.remove()
        logger.add(
            sink or sys.stdout,
            level=level,
            format=log_format,
            enqueue=enqueue,
            backtrace=False,
            diagnose=False,
        )
        logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)

else:

    logger = logging.getLogger("paper_collector")
    logger.propagate = False

    def setup_logging(
        level: str = "INFO",
        sink=None,
        log_format: str | None = None,
        enqueue: bool = True,  # kept for signature parity
    ) -> None:
        """Fallback: standard logging to stdout when loguru is unavailable."""

        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        if sink is None:
            handler = logging.StreamHandler(sys.stdout)
        elif hasattr(sink, "write"):
            handler = logging.StreamHandler(sink)
        else:
            class _FuncHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    msg = self.format(record)
                    sink(msg)

            handler = _FuncHandler()

        formatter = logging.Formatter(
            log_format
            or "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.handlers.clear()
        logger.addHandler(handler)

        logging.basicConfig(level=numeric_level, handlers=[handler], force=True)


setup_logging()

__all__ = ["logger", "setup_logging", "PLAIN_LOG_FORMAT"]
