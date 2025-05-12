import os
import sys

from loguru import logger
from rich.logging import RichHandler

# Import the centralized settings loader and type hint
from startup_researcher.config import AppSettings, get_settings

# Remove direct yaml import and DEFAULT_LOGGING_CONFIG, load_config function
# They are replaced by the pydantic-settings mechanism in config.py


# --- Logger Setup ---
def setup_logging(app_settings: AppSettings):  # Pass settings explicitly
    """Sets up the logger based on the Pydantic settings."""
    cfg = app_settings.logging  # Access the logging section directly

    logger.remove()  # Remove default handler

    base_log_path = cfg.base_log_path
    os.makedirs(base_log_path, exist_ok=True)

    # Console Handler (Rich)
    console_cfg = cfg.console  # Access sub-model
    if console_cfg.enabled:
        # Use attributes directly from the Pydantic model
        logger.add(
            RichHandler(
                level=console_cfg.level.upper(),
                rich_tracebacks=console_cfg.rich_tracebacks,
                tracebacks_show_locals=console_cfg.tracebacks_show_locals,
                tracebacks_word_wrap=console_cfg.tracebacks_word_wrap,
                markup=console_cfg.markup,
                show_time=console_cfg.show_time,
                show_level=console_cfg.show_level,
                show_path=console_cfg.show_path,
                log_time_format=console_cfg.log_time_format),
            level=console_cfg.level.upper(),
            format="{message}",  # RichHandler handles its own formatting
            enqueue=False,  # Sensible default for console
            backtrace=console_cfg.
            rich_tracebacks,  # Align with RichHandler setting
            diagnose=
            False  # Typically False for console, can be configurable if needed
        )
        logger.info(
            f"Console logging enabled at level {console_cfg.level.upper()}.")

    # File Handler
    file_cfg = cfg.file  # Access sub-model
    if file_cfg.enabled:
        log_file_path = os.path.join(base_log_path, file_cfg.log_name)
        logger.add(
            log_file_path,
            level=file_cfg.level.upper(),
            format=file_cfg.format,
            rotation=file_cfg.rotation,
            retention=file_cfg.retention,
            compression=file_cfg.compression,
            backtrace=file_cfg.backtrace,
            diagnose=file_cfg.diagnose,
            enqueue=file_cfg.enqueue,
            delay=file_cfg.delay,
            encoding='utf-8'  # Explicitly set encoding for file sinks
        )
        logger.info(f"File logging to '{log_file_path}' enabled "
                    f"at level {file_cfg.level.upper()}.")

    # Structured File Handler (JSONL)
    structured_cfg = cfg.structured_file  # Access sub-model
    if structured_cfg.enabled:
        structured_log_file_path = os.path.join(base_log_path,
                                                structured_cfg.log_name)
        logger.add(
            structured_log_file_path,
            level=structured_cfg.level.upper(),
            rotation=structured_cfg.rotation,
            retention=structured_cfg.retention,
            compression=structured_cfg.compression,
            serialize=True,  # Key for JSON output
            backtrace=structured_cfg.backtrace,
            diagnose=structured_cfg.diagnose,
            enqueue=structured_cfg.enqueue,
            delay=structured_cfg.delay,
            encoding='utf-8'  # Explicitly set encoding for file sinks
        )
        logger.info(f"Structured file logging to '{structured_log_file_path}' "
                    f"enabled at level {structured_cfg.level.upper()}.")

    if not console_cfg.enabled and \
       not file_cfg.enabled and \
       not structured_cfg.enabled:
        logger.warning("All logging handlers are disabled in configuration. "
                       "No logs will be output.")
        # Add a basic handler to stderr to avoid Loguru's default handler
        # if nothing else is enabled, preventing potential errors.
        logger.add(sys.stderr,
                   level="WARNING",
                   format="{time} | {level} | {message}")


# Call setup_logging() when the module is imported so logger is ready.
# Load settings from the central config module
settings = get_settings()
setup_logging(settings)


def get_logger():
    """Returns the configured Loguru logger instance."""
    return logger


# Example usage (primarily for testing this module directly):
if __name__ == "__main__":
    # Settings are already loaded by config.py
    # We can access them directly if needed for the example.
    current_settings = get_settings()
    log = get_logger()  # Logger is already configured

    log.info("Testing logger configured via Pydantic Settings...")

    log.trace("This is a TRACE message.")
    log.debug("This is a DEBUG message with {'key': 'value'}.")
    log.info("This is an INFO message. E.g., a startup name: "
             "Startup Y http://example-y.com")
    log.success("This is a SUCCESS message.")
    log.warning("This is a WARNING message.")
    log.error("This is an ERROR message.")
    log.critical("This is a CRITICAL message.")

    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("A caught ZeroDivisionError occurred.")

    log.info("Logging setup complete. Check console and configured log files.")

    # Access config via settings object for the example output
    if current_settings.logging.file.enabled:
        log_path = os.path.join(current_settings.logging.base_log_path,
                                current_settings.logging.file.log_name)
        log.info(f"File log configured at: {log_path}")
    if current_settings.logging.structured_file.enabled:
        log_path = os.path.join(
            current_settings.logging.base_log_path,
            current_settings.logging.structured_file.log_name)
        log.info(f"Structured log configured at: {log_path}")
