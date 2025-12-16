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

    def setup_logging(level: str = "INFO") -> None:
        """Initialize loguru with stdout sink and intercept stdlib logging."""
        logger.remove()
        logger.add(
            sys.stdout,
            level=level,
            format=LOG_FORMAT,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
        logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)

else:

    logger = logging.getLogger("paper_collector")
    logger.propagate = False

    def setup_logging(level: str = "INFO") -> None:
        """Fallback: standard logging to stdout when loguru is unavailable."""

        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.handlers.clear()
        logger.addHandler(handler)

        logging.basicConfig(level=numeric_level, handlers=[handler], force=True)


setup_logging()

__all__ = ["logger", "setup_logging"]
