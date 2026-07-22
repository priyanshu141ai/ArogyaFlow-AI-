from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class ApplicationMetrics:
    def __init__(self) -> None:
        self._registry = CollectorRegistry()
        self.requests = Counter(
            "arogyaflow_http_requests_total",
            "HTTP requests handled by the API.",
            ("method", "route", "status"),
            registry=self._registry,
        )
        self.request_duration = Histogram(
            "arogyaflow_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("method", "route"),
            registry=self._registry,
        )
        self.database_ready = Gauge(
            "arogyaflow_database_ready",
            "Whether the configured database passed its latest health check.",
            registry=self._registry,
        )
        self.model_available = Gauge(
            "arogyaflow_model_artifact_available",
            "Whether a configured model artifact exists.",
            ("model",),
            registry=self._registry,
        )

    def observe_request(self, method: str, route: str, status_code: int, duration: float) -> None:
        self.requests.labels(method, route, str(status_code)).inc()
        self.request_duration.labels(method, route).observe(duration)

    def render(self) -> bytes:
        return generate_latest(self._registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST
