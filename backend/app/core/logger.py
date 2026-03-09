import logging
import sys
from pythonjsonlogger import jsonlogger
from pathlib import Path

def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Настройка логирования для приложения

    Args:
        debug: Если True, уровень логирования DEBUG, иначе INFO.
    Returns:
        Класс Logger, настроенный для проекта.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("Banking77")
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if not debug:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%T%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%T%H:%M:%S"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    logger.info(f"Logging initialized | level={logging.getLevelName(log_level)}")

    return logger
