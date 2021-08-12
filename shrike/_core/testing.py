# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utility methods around unit testing.
"""

import io
import logging
from typing import Union


class StreamHandlerContext:
    """
    Add, then remove a stream handler with the provided format string. The
    `__str__` method on this class returns the value of the internal stream.
    """

    def __init__(self, log: logging.Logger, fmt: str, level):
        self.logger = log

        self._original_level = log.level
        # It's important to set the logger level before adding the handler.
        self.logger.setLevel(level)

        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter(fmt))
        self.handler.setLevel(level)

    def __enter__(self):
        self.logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.removeHandler(self.handler)
        self.handler.flush()
        self.logger.setLevel(self._original_level)

    def __str__(self):
        return self.stream.getvalue()


def stream_handler(
    log: Union[str, logging.Logger],
    format: str = "%(levelname)s:%(name)s:%(message)s",
    level: Union[None, int, str] = None,
) -> StreamHandlerContext:
    """
    Attach an in-memory stream handler to the provided logger, either a Python
    object or the name of the logger. Can optionally provide log format and
    level.
    """
    if isinstance(log, str):
        log = logging.getLogger(log)

    if not level:
        level = log.getEffectiveLevel()

    return StreamHandlerContext(log, format, level)
