import os
import sys  # Import sys for stderr printing on critical error
from typing import List, Optional, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# --- Pydantic Models for Configuration ---


class ConsoleLoggingSettings(BaseModel):
    enabled: bool = True
    level: str = 'DEBUG'
    rich_tracebacks: bool = True
    tracebacks_show_locals: bool = False
    tracebacks_word_wrap: bool = True
    markup: bool = True
    show_time: bool = True
    show_level: bool = True
    show_path: bool = True
    log_time_format: str = '[%X]'  # HH:MM:SS


class FileLoggingSettings(BaseModel):
    enabled: bool = False
    level: str = 'INFO'
    log_name: str = 'app.log'
    rotation: str = '100 MB'
    retention: str = '10 days'
    compression: Optional[str] = None
    format: str = ("{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                   "{name}:{function}:{line} - {message}{exception}")
    backtrace: bool = True
    diagnose: bool = True
    enqueue: bool = False
    delay: bool = False


class StructuredFileLoggingSettings(BaseModel):
    enabled: bool = False
    level: str = 'INFO'
    log_name: str = 'app_structured.jsonl'
    rotation: str = '100 MB'
    retention: str = '10 days'
    compression: Optional[str] = None
    backtrace: bool = True
    diagnose: bool = True
    enqueue: bool = False
    delay: bool = False


class LoggingSettings(BaseModel):
    """Defines the 'logging' section of the config."""
    base_log_path: str = 'logs'
    console: ConsoleLoggingSettings = ConsoleLoggingSettings()
    file: FileLoggingSettings = FileLoggingSettings()
    structured_file: StructuredFileLoggingSettings = (
        StructuredFileLoggingSettings())


class AppSettings(BaseSettings):
    """Main application settings model."""
    # Add other top-level config sections here if needed
    startups: List[str] = []  # Added startups based on docs.md
    logging: LoggingSettings = LoggingSettings()

    model_config = SettingsConfigDict(
        # Specifies the YAML file to load settings from.
        # Assumes config.yaml is in the current working directory when run.
        # In a real app, you might want a more robust path discovery.
        yaml_file='config.yaml',
        # extra='ignore',  # Ensure this line is REMOVED or commented out
        # Add other pydantic-settings configurations if needed:
        # env_prefix = 'APP_' # Example: Prefix for environment variables
        # case_sensitive = False
        # env_nested_delimiter = '__'
    )

    # Customise sources to prioritize YAML file loading
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Default order: init -> env -> dotenv -> secrets
        # We add YamlConfigSettingsSource to load from YAML file.
        # Place YAML loading before default values are applied.
        # You might adjust the order based on desired priority
        # (e.g., env vars overriding YAML).
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls),  # Explicitly add YAML source
            file_secret_settings,
            # Note: Default values defined in models act as the last source.
        )


# --- Singleton Instance ---
# Load settings once when the module is imported
# Handle potential errors during loading gracefully
try:
    settings = AppSettings()
except Exception as e:
    # Use basic print/stderr as logger might not be configured yet
    print(
        f"CRITICAL: Failed to load application settings from config.yaml: {e}",
        file=sys.stderr)
    # Fallback to default settings if loading fails completely
    settings = AppSettings()  # This will use default values defined in models


def get_settings() -> AppSettings:
    """Returns the loaded application settings instance."""
    # In a real app, consider potential reloads or dynamic updates if needed
    return settings


# Example usage (primarily for testing this module directly):
if __name__ == "__main__":
    print("Loaded Application Settings:")
    # Use model_dump_json for clean output
    print(settings.model_dump_json(indent=2))

    # Example: Accessing a specific setting
    print(f"\nConsole logging enabled: {settings.logging.console.enabled}")
    log_file_name = (settings.logging.file.log_name
                     if settings.logging.file else 'N/A')
    print(f"Log file path: "
          f"{os.path.join(settings.logging.base_log_path, log_file_name)}")
    print(f"Startups to research: {settings.startups}")
