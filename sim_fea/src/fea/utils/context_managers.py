from collections.abc import Iterator
from contextlib import contextmanager

import gmsh  # pyright: ignore[reportMissingTypeStubs]
from loguru import logger


@contextmanager
def gmsh_session() -> Iterator[None]:
    """
    Context manager to handle Gmsh initialisation and finalisation
    """
    try:
        logger.info("Attempting to initialise Gmsh...")
        gmsh.initialize()  # pyright: ignore[reportUnknownMemberType]
        yield gmsh
    except RuntimeError as e:
        logger.error(f"Gmsh internal error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occured: {e}")
        raise
    finally:
        if gmsh.isInitialized():
            logger.info("Finalising Gmsh...")
            gmsh.finalize()
