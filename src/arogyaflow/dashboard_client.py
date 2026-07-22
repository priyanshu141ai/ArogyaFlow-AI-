from typing import Any

import httpx2 as httpx

from arogyaflow.exceptions import DashboardApiError


class DashboardClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
                response = client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise DashboardApiError("ArogyaFlow API is unavailable") from exc
        if not response.is_success:
            try:
                error = response.json()
            except ValueError:
                error = {}
            if not isinstance(error, dict):
                error = {}
            message = error.get("message", f"API request failed with {response.status_code}")
            raise DashboardApiError(str(message))
        data = response.json()
        if not isinstance(data, dict):
            raise DashboardApiError("API returned an invalid response")
        return data

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)
