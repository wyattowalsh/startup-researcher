import sys
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger as loguru_logger  # Keep original logger for assertions
from rich.logging import RichHandler

from startup_researcher.config import (
    AppSettings,
    ConsoleLoggingSettings,
    FileLoggingSettings,
    LoggingSettings,
    StructuredFileLoggingSettings,
)
from startup_researcher.logging import get_logger, setup_logging

# --- Fixtures ---


@pytest.fixture
def mock_loguru():
    # Patch the logger instance in the logging module
    with patch('startup_researcher.logging.logger') as mock_logger:
        # Reset mocks for each test to avoid state leakage
        mock_logger.reset_mock()
        mock_logger.add = MagicMock()
        mock_logger.remove = MagicMock()
        mock_logger.info = MagicMock()
        mock_logger.warning = MagicMock()
        yield mock_logger


@pytest.fixture
def mock_rich_handler():
    with patch('startup_researcher.logging.RichHandler') as mock_rich:
        mock_rich.return_value = MagicMock(spec=RichHandler)
        yield mock_rich


@pytest.fixture
def mock_os_makedirs():
    with patch('startup_researcher.logging.os.makedirs') as mock_makedirs:
        yield mock_makedirs


# Fixture providing settings reflecting Pydantic *model* defaults
@pytest.fixture
def model_default_settings():
    """Provides AppSettings reflecting model defaults (no YAML loaded)."""
    # Directly instantiate models to bypass BaseSettings loading mechanism
    return AppSettings(
        logging=LoggingSettings(  # Explicitly use default sub-models
            console=ConsoleLoggingSettings(),
            file=FileLoggingSettings(),
            structured_file=StructuredFileLoggingSettings()),
        startups=[])


@pytest.fixture
def minimal_settings():
    """Provides settings with all loggers disabled."""
    return AppSettings(
        logging=LoggingSettings(console=ConsoleLoggingSettings(enabled=False),
                                file=FileLoggingSettings(enabled=False),
                                structured_file=StructuredFileLoggingSettings(
                                    enabled=False)))


# --- Test Cases ---


def test_setup_logging_model_defaults(
        model_default_settings,  # Use the fixture for model defaults
        mock_loguru,
        mock_rich_handler,
        mock_os_makedirs):
    """Test setup with Pydantic model defaults (console True, others False)."""
    setup_logging(model_default_settings)  # Pass the correct settings

    mock_loguru.remove.assert_called_once()
    mock_os_makedirs.assert_called_once_with(
        model_default_settings.logging.base_log_path, exist_ok=True)

    # Check RichHandler (console) was configured and added
    mock_rich_handler.assert_called_once_with(
        level=model_default_settings.logging.console.level.upper(),
        rich_tracebacks=model_default_settings.logging.console.rich_tracebacks,
        tracebacks_show_locals=model_default_settings.logging.console.
        tracebacks_show_locals,
        tracebacks_word_wrap=model_default_settings.logging.console.
        tracebacks_word_wrap,
        markup=model_default_settings.logging.console.markup,
        show_time=model_default_settings.logging.console.show_time,
        show_level=model_default_settings.logging.console.show_level,
        show_path=model_default_settings.logging.console.show_path,
        log_time_format=model_default_settings.logging.console.log_time_format)
    # With model defaults, only console is enabled
    assert mock_loguru.add.call_count == 1  # Should be 1 now
    mock_loguru.add.assert_called_once_with(
        mock_rich_handler.return_value,
        level=model_default_settings.logging.console.level.upper(),
        format="{message}",
        enqueue=False,
        backtrace=model_default_settings.logging.console.rich_tracebacks,
        diagnose=False)
    # Check info message was logged for console
    mock_loguru.info.assert_called_once()
    assert "Console logging" in mock_loguru.info.call_args[0][0]


def test_setup_logging_all_enabled(
        model_default_settings,  # Use base settings
        mock_loguru,
        mock_rich_handler,
        mock_os_makedirs):
    """Test setup when console, file, and structured logs are enabled."""
    # Modify a copy of the model defaults
    settings = model_default_settings.model_copy(deep=True)  # Use model_copy
    settings.logging.file.enabled = True
    settings.logging.structured_file.enabled = True
    settings.logging.file.log_name = "test_file.log"
    settings.logging.structured_file.log_name = "test_structured.jsonl"

    setup_logging(settings)

    mock_loguru.remove.assert_called_once()
    mock_os_makedirs.assert_called_once_with(settings.logging.base_log_path,
                                             exist_ok=True)

    # Expected paths
    expected_file_path = (
        f"{settings.logging.base_log_path}/{settings.logging.file.log_name}")
    expected_structured_path = (f"{settings.logging.base_log_path}/"
                                f"{settings.logging.structured_file.log_name}")

    # Check all three handlers were added
    assert mock_loguru.add.call_count == 3

    calls = mock_loguru.add.call_args_list

    # Console (Rich)
    console_call = next(
        (c for c in calls if c.args[0] == mock_rich_handler.return_value),
        None)
    assert console_call is not None
    assert console_call.kwargs[
        'level'] == settings.logging.console.level.upper()

    # File
    file_call = next((c for c in calls if c.args[0] == expected_file_path),
                     None)
    assert file_call is not None
    assert file_call.kwargs['level'] == settings.logging.file.level.upper()
    assert file_call.kwargs['rotation'] == settings.logging.file.rotation
    assert file_call.kwargs['retention'] == settings.logging.file.retention
    assert file_call.kwargs['compression'] == settings.logging.file.compression
    assert file_call.kwargs['format'] == settings.logging.file.format
    # assert file_call.kwargs['serialize'] is False # REMOVED: Not applicable
    assert file_call.kwargs['encoding'] == 'utf-8'

    # Structured File
    structured_call = next(
        (c for c in calls if c.args[0] == expected_structured_path), None)
    assert structured_call is not None
    assert structured_call.kwargs['level'] == \
        settings.logging.structured_file.level.upper()
    assert structured_call.kwargs['rotation'] == \
        settings.logging.structured_file.rotation
    assert structured_call.kwargs['serialize'] is True  # Correctly checked here
    assert structured_call.kwargs['encoding'] == 'utf-8'

    # Check info messages
    assert mock_loguru.info.call_count == 3


def test_setup_logging_none_enabled(minimal_settings, mock_loguru,
                                    mock_os_makedirs):
    """Test setup when all handlers are disabled, expects fallback handler."""
    setup_logging(minimal_settings)

    mock_loguru.remove.assert_called_once()
    mock_os_makedirs.assert_called_once_with(
        minimal_settings.logging.base_log_path, exist_ok=True)

    # Check warning message was logged
    mock_loguru.warning.assert_called_once()
    assert "All logging handlers are disabled" in \
        mock_loguru.warning.call_args[0][0]

    # Check fallback handler was added to stderr
    assert mock_loguru.add.call_count == 1
    mock_loguru.add.assert_called_once_with(
        sys.stderr, level="WARNING", format="{time} | {level} | {message}")


def test_get_logger():
    """Test that get_logger returns the loguru logger instance."""
    # Need to ensure logging is imported freshly potentially
    # But get_logger just returns the module-level logger instance
    from startup_researcher.logging import logger as imported_logger
    logger_instance = get_logger()
    assert logger_instance is imported_logger
    assert logger_instance is loguru_logger  # Check it's the loguru one
