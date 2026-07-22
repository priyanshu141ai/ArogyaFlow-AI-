from collections import deque
from secrets import compare_digest
from threading import Lock
from time import monotonic

PROTECTED_PATHS = ("/v1/", "/metrics", "/docs", "/redoc", "/openapi.json")


def is_protected_path(path: str) -> bool:
    return path in PROTECTED_PATHS or any(path.startswith(prefix) for prefix in PROTECTED_PATHS)


def valid_api_key(expected: str | None, provided: str | None) -> bool:
    return expected is None or (provided is not None and compare_digest(expected, provided))


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60, max_clients: int = 10_000) -> None:
        if limit < 1 or window_seconds <= 0 or max_clients < 1:
            raise ValueError("Rate limiter values must be positive")
        self._limit = limit
        self._window_seconds = window_seconds
        self._max_clients = max_clients
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        observed_at = monotonic() if now is None else now
        cutoff = observed_at - self._window_seconds
        with self._lock:
            if client not in self._requests and len(self._requests) >= self._max_clients:
                self._purge(cutoff)
                if len(self._requests) >= self._max_clients:
                    return False
            requests = self._requests.setdefault(client, deque())
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self._limit:
                return False
            requests.append(observed_at)
            return True

    def _purge(self, cutoff: float) -> None:
        for client, requests in tuple(self._requests.items()):
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if not requests:
                del self._requests[client]
