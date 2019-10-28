import logging


def set_logging_level(logging_level):
    logging_level = logging_level.lower()

    if logging_level == "critical":
        level = logging.CRITICAL
    elif logging_level == "warning":
        level = logging.WARNING
    elif logging_level == "info":
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        format='%(asctime)s [%(filename)s]: %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=level
    )
