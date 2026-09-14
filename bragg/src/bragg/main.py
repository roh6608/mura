"""Entry point for the Bragg-edge fitting application."""

from loguru import logger

from bragg.core.config import load_config
from bragg.core.process_manager import BraggProcessManager
from bragg.utils.arg_parser import parse_args


def main() -> None:
    """Run the Bragg-edge fit described by the configuration file."""
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    manager = BraggProcessManager(config)
    manager.start()

    try:
        manager.bragg.join()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
