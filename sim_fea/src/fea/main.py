from loguru import logger

from fea.core.config import load_config
from fea.core.process_manager import FeaProcessManager
from fea.utils.arg_parser import parse_args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    manager = FeaProcessManager(config)
    manager.start()

    try:
        manager.solver.join()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
