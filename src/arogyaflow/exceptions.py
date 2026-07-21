class ArogyaFlowError(Exception):
    """Base exception for known domain failures."""


class ConfigurationError(ArogyaFlowError):
    """Raised when application configuration is unusable."""


class DataContractError(ArogyaFlowError):
    """Raised when a dataset does not match its declared schema."""


class DataQualityError(ArogyaFlowError):
    """Raised when data cannot be safely validated or quarantined."""


class FeatureLeakageError(ArogyaFlowError):
    """Raised when evaluation uses information unavailable at prediction time."""


class ValidationError(ArogyaFlowError):
    """Raised when domain input is invalid."""


class ResourceNotFoundError(ArogyaFlowError):
    """Raised when a requested domain resource does not exist."""
