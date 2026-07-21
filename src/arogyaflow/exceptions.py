class ArogyaFlowError(Exception):
    """Base exception for known domain failures."""


class ConfigurationError(ArogyaFlowError):
    """Raised when application configuration is unusable."""


class ValidationError(ArogyaFlowError):
    """Raised when domain input is invalid."""


class ResourceNotFoundError(ArogyaFlowError):
    """Raised when a requested domain resource does not exist."""
