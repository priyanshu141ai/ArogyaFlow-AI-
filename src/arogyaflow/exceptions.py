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


class FeatureSchemaMismatchError(ArogyaFlowError):
    """Raised when inference features do not match the trained schema."""


class TrainingDataError(ArogyaFlowError):
    """Raised when data cannot safely produce an accepted model."""


class ModelArtifactError(ArogyaFlowError):
    """Raised when a persisted model artifact is invalid."""


class ForecastHorizonError(ArogyaFlowError):
    """Raised when a forecast horizon is unsupported."""


class SimulationConfigurationError(ArogyaFlowError):
    """Raised when a simulation configuration is not operationally valid."""


class RecommendationConstraintError(ArogyaFlowError):
    """Raised when a recommendation violates an explicit constraint."""


class PersistenceError(ArogyaFlowError):
    """Raised when an operational record cannot be safely persisted."""


class DashboardApiError(ArogyaFlowError):
    """Raised when the dashboard cannot obtain a valid API response."""


class ExperimentTrackingError(ArogyaFlowError):
    """Raised when an MLflow run or registry update fails."""


class DemoSetupError(ArogyaFlowError):
    """Raised when the local demo stack cannot be prepared or started."""


class ValidationError(ArogyaFlowError):
    """Raised when domain input is invalid."""


class ResourceNotFoundError(ArogyaFlowError):
    """Raised when a requested domain resource does not exist."""
