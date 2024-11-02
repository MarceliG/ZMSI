import logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s() | %(message)s",  # Include filename and function name
    datefmt="%m-%d-%Y %H:%M:%S",
)

logger = logging.getLogger()
