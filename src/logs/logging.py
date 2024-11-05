import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s() | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)

logger = logging.getLogger()
