"""Entry point for the pre-processing application."""

from loguru import logger

from preprocess.core.config import load_config
from preprocess.core.process_manager import PreprocessProcessManager
from preprocess.utils.arg_parser import parse_args


def main() -> None:
    """Run the pre-processing chain described by the configuration file."""
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    manager = PreprocessProcessManager(config)
    manager.start()

    try:
        manager.preprocess.join()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
