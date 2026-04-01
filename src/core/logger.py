from loguru import logger


def setup_logger(level: str = "INFO"):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    return logger
