import logging

# Set up a generic logger for the memor library.
# We attach a NullHandler so that by default, the library is completely silent.
# Downstream applications (like chat.py) can attach their own handlers
# (e.g., StreamHandler) and set the log level to INFO or DEBUG to see the logs.
logger = logging.getLogger("memor")
logger.addHandler(logging.NullHandler())

from .engine import MemoryEngine

__version__ = "0.2.0"
__all__ = ["MemoryEngine", "__version__"]
