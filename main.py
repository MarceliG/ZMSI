from src.logs import logger
from src.preprocessing import analyses


def main() -> None:
    """Execute The main function."""
    logger.info("Start application")

    analyses()


if __name__ == "__main__":
    main()
