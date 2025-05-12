import pytest
import yaml
from pydantic import ValidationError

from startup_researcher.config import (
    AppSettings,
    ConsoleLoggingSettings,
    FileLoggingSettings,
    LoggingSettings,
    StructuredFileLoggingSettings,
    get_settings,
)


# Helper fixture that returns a factory function
@pytest.fixture
def config_factory(tmp_path):

    def _create_dummy_config(content):
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(content, f)
        return str(config_path)

    return _create_dummy_config


# Test loading default settings when no config file exists
def test_load_defaults_no_file(monkeypatch):
    # Ensure no real config.yaml is loaded
    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': 'non_existent_config.yaml'})
    # Reload the settings by creating a new instance (as it loads on import)
    # We need to mock the constructor or the source loading
    # Easier to test the structure of defaults directly
    settings = AppSettings()  # Uses defaults defined in the model

    assert isinstance(settings.logging, LoggingSettings)
    assert isinstance(settings.logging.console, ConsoleLoggingSettings)
    assert isinstance(settings.logging.file, FileLoggingSettings)
    assert isinstance(settings.logging.structured_file,
                      StructuredFileLoggingSettings)

    # Check some default values
    assert settings.logging.console.enabled is True
    assert settings.logging.console.level == 'DEBUG'
    assert settings.logging.file.enabled is False
    assert settings.logging.file.log_name == 'app.log'
    assert settings.logging.structured_file.enabled is False
    assert settings.startups == []


# Test loading settings from a basic YAML file
def test_load_from_yaml(config_factory, monkeypatch):
    config_content = {
        'logging': {
            'base_log_path': '/tmp/logs',
            'console': {
                'level': 'INFO'
            },
            'file': {
                'enabled': True,
                'log_name': 'myapp.log'
            }
        },
        'startups': ['Startup A', 'Startup B']
    }
    config_path = config_factory(config_content)

    # Patch the model_config to use the temp file
    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': config_path})

    # Re-instantiate to trigger loading from the new path
    settings = AppSettings()

    assert settings.logging.base_log_path == '/tmp/logs'
    assert settings.logging.console.level == 'INFO'
    assert settings.logging.console.enabled is True  # Default kept
    assert settings.logging.file.enabled is True
    assert settings.logging.file.log_name == 'myapp.log'
    assert settings.logging.file.level == 'INFO'  # Default kept
    assert settings.logging.structured_file.enabled is False  # Default kept
    assert settings.startups == ['Startup A', 'Startup B']


# Test overriding settings with environment variables
def test_override_with_env_vars(config_factory, monkeypatch):
    config_content = {'logging': {'console': {'level': 'INFO'}}}
    config_path = config_factory(config_content)

    # Set environment variables (use a prefix if defined in model_config)
    # Assuming no env_prefix for these tests
    monkeypatch.setenv("LOGGING__CONSOLE__LEVEL", "WARNING")
    monkeypatch.setenv("LOGGING__FILE__ENABLED", "true")
    monkeypatch.setenv("LOGGING__FILE__LOG_NAME", "env_override.log")
    monkeypatch.setenv("STARTUPS", '["Env Startup"]')  # JSON encoded list

    # Patch the model_config to use the temp file and add env delimiter
    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config', {
        'yaml_file': config_path,
        'env_nested_delimiter': '__'
    })

    # Re-instantiate to trigger loading
    settings = AppSettings()

    # Env vars should override YAML/defaults
    assert settings.logging.console.level == 'WARNING'
    assert settings.logging.file.enabled is True
    assert settings.logging.file.log_name == 'env_override.log'
    assert settings.startups == ["Env Startup"]


# Test handling of invalid YAML structure (missing logging section)
def test_load_invalid_yaml_structure(config_factory, monkeypatch):
    config_content = {
        'other_section': 'some_value'
        # 'logging': {} # Missing logging section, but also an extra field
    }
    config_path = config_factory(config_content)

    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': config_path})

    # With default strict behavior (extra='forbid'), this should raise ValidationError
    with pytest.raises(ValidationError) as excinfo:
        AppSettings()

    # Verify that the error is due to the extra field
    assert "other_section" in str(excinfo.value).lower()
    assert "extra inputs are not permitted" in str(excinfo.value).lower()


# Test handling of YAML syntax error
def test_yaml_syntax_error(monkeypatch):

    def mock_yaml_source(*args, **kwargs):
        raise yaml.YAMLError("Simulated YAML syntax error")

    monkeypatch.setattr('startup_researcher.config.YamlConfigSettingsSource',
                        mock_yaml_source)

    # Instantiating AppSettings should raise the underlying error from source
    with pytest.raises(yaml.YAMLError, match="Simulated YAML syntax error"):
        AppSettings()


# Test validation error for incorrect type in config
def test_validation_error_from_yaml(config_factory, monkeypatch):
    config_content = {'logging': {'console': {'enabled': 'not-a-boolean'}}}
    config_path = config_factory(config_content)  # Use the factory

    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': config_path})

    with pytest.raises(ValidationError):
        AppSettings()


# Test get_settings returns the loaded instance
def test_get_settings(monkeypatch):
    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': 'non_existent_config.yaml'})
    loaded_settings = AppSettings()
    monkeypatch.setattr('startup_researcher.config.settings', loaded_settings)

    retrieved_settings = get_settings()
    assert retrieved_settings is loaded_settings
    assert retrieved_settings.logging.console.level == 'DEBUG'


# Test default values within nested models
def test_nested_model_defaults(monkeypatch):  # Added monkeypatch
    # Ensure no real config.yaml is loaded to test Pydantic model defaults
    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': 'non_existent_config.yaml'})
    settings = AppSettings()
    assert settings.logging.console.rich_tracebacks is True
    assert settings.logging.console.tracebacks_show_locals is False
    assert settings.logging.file.rotation == '100 MB'
    assert settings.logging.file.compression is None  # This should now pass
    assert settings.logging.structured_file.log_name == 'app_structured.jsonl'
    assert settings.logging.structured_file.backtrace is True


# Test handling of invalid YAML structure (e.g., empty file)
def test_load_empty_yaml_structure(config_factory, monkeypatch):
    config_content = {}  # Use an empty config
    config_path = config_factory(config_content)

    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': config_path})

    # Instantiate - should use defaults
    settings = AppSettings()

    # Check defaults are used
    assert settings.logging.console.enabled is True
    assert settings.logging.console.level == 'DEBUG'
    assert settings.startups == []


# Test customization of sources priority
# Implicitly tested by test_load_from_yaml, test_override_with_env_vars.
# Reformatting comment to fit line length.
# A more direct test could mock sources.


# Test that loading YAML with extra, undefined fields raises a ValidationError
def test_load_yaml_with_extra_fields_raises_error(config_factory, monkeypatch):
    config_content = {
        'logging': {  # Valid section
            'console': {
                'level': 'INFO'
            }
        },
        'extra_top_level_field': 'some_value',  # Invalid extra field
        'another_extra': 123
    }
    config_path = config_factory(config_content)

    monkeypatch.setattr('startup_researcher.config.AppSettings.model_config',
                        {'yaml_file': config_path})

    with pytest.raises(ValidationError) as excinfo:
        AppSettings()

    # Optionally, check the error messages for more specific details
    # For example, check that 'extra_top_level_field' is mentioned
    assert "extra_top_level_field" in str(excinfo.value).lower()
    assert "another_extra" in str(excinfo.value).lower()
    assert "extra inputs are not permitted" in str(excinfo.value).lower()
