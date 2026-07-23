"""Colored, level-aware logging for training / inference / benchmarking.

A thin wrapper over the stdlib ``logging`` module that gives:
- a compact, aligned console format ``HH:MM:SS  LEVEL  name  message``,
- per-severity colors (INFO/OK/WARN/ERROR), auto-disabled when stdout is not a
  TTY (so redirected logs stay clean),
- an ANSI-free file log for the ``Train`` logger,
- a custom ``SUCCESS`` level, and convenience methods (``info``/``success``/
  ``warning``/``error``/``debug``), while keeping the legacy
  ``log_with_color(message)`` entry point working.
"""
import logging
import os
import sys

from colorama import Fore, Style, init as _colorama_init

_colorama_init(autoreset=False)

# A success level sitting just above INFO (and below WARNING).
SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

_LEVEL_COLOR = {
    logging.DEBUG: Style.DIM + Fore.WHITE,
    logging.INFO: Fore.BLUE,
    SUCCESS: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Style.BRIGHT + Fore.RED,
}
_LEVEL_LABEL = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    SUCCESS: "OK",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT",
}


class _ColorFormatter(logging.Formatter):
    """Console formatter: dim timestamp, colored level tag + name, message
    (message tinted only for WARNING and above, and for SUCCESS, so routine
    INFO lines stay readable)."""

    def __init__(self, use_color=True):
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        label = _LEVEL_LABEL.get(record.levelno, record.levelname)
        msg = record.getMessage()
        if not self.use_color:
            return f"{ts} | {label:<5} | {record.name} | {msg}"
        color = _LEVEL_COLOR.get(record.levelno, "")
        reset = Style.RESET_ALL
        tint = color if (record.levelno >= logging.WARNING or record.levelno == SUCCESS) else ""
        return (
            f"{Style.DIM}{ts}{reset} "
            f"{color}{label:<5}{reset} "
            f"{Style.DIM}{record.name}{reset}  "
            f"{tint}{msg}{reset if tint else ''}"
        )


def _console_supports_color():
    if os.environ.get("RFUAV_LOG_COLOR") == "0":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class colorful_logger:
    """Small logging facade used across training/inference/benchmark.

    Parameters:
    - name (str): logger name shown in each line (e.g. ``Train``, ``Inference``).
    - logfile (str, optional): if given, also append plain (ANSI-free) lines to
      this file.
    """

    def __init__(self, name, logfile=None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.propagate = False  # don't double-log via the root logger

        if not self.logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(_ColorFormatter(use_color=_console_supports_color()))
            self.logger.addHandler(console)

            if logfile:
                os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
                file_handler = logging.FileHandler(logfile, encoding="utf-8")
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
                )
                self.logger.addHandler(file_handler)

            self.logger.setLevel(logging.INFO)

        self._deduplicate_handlers()

    def _deduplicate_handlers(self):
        seen = {}
        for handler in list(self.logger.handlers):
            key = (type(handler), getattr(handler, "baseFilename", None))
            if key in seen:
                self.logger.removeHandler(handler)
            else:
                seen[key] = handler

    # --- level-aware API ---------------------------------------------------
    def info(self, message):
        self.logger.info(message)

    def success(self, message):
        self.logger.log(SUCCESS, message)

    def warning(self, message):
        self.logger.warning(message)

    warn = warning

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

    # --- legacy entry point (kept for existing call sites) -----------------
    def log_with_color(self, message=None, color=None, level=logging.INFO):
        """Log at ``level`` (INFO by default). Coloring is driven by the level;
        an explicit colorama ``color`` is still honored if passed."""
        if color is not None:
            message = f"{color}{message}{Style.RESET_ALL}"
        self.logger.log(level, message)
