"""Minimal JSON-RPC client for interacting with a Sui fullnode."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

try:  # pragma: no cover - import guard for clearer error messaging
    import requests
    from requests import RequestException, Session
except ImportError as exc:  # pragma: no cover - raises during module import
    raise ImportError(
        "The 'requests' package is required for Sui RPC interactions. "
        "Install it with 'pip install requests'."
    ) from exc


class SuiRPCError(RuntimeError):
    """Raised when the Sui node returns an error response."""

    def __init__(self, method: str, error: Dict[str, Any]):
        super().__init__(f"RPC call {method} failed: {error}")
        self.method = method
        self.error = error


@dataclass(slots=True)
class SuiRPCConfig:
    url: str
    timeout: float = 30.0
    verify: bool = True
    cafile: Optional[str] = None


class SuiRPCClient:
    """Tiny JSON-RPC client for Sui fullnode endpoints."""

    def __init__(
        self,
        config: SuiRPCConfig | None = None,
        *,
        url: str | None = None,
        timeout: float = 30.0,
        verify: bool = True,
        cafile: Optional[str] = None,
    ) -> None:
        if config is None and url is None:
            raise ValueError("Provide either config or url")
        if config is None:
            config = SuiRPCConfig(url=url or "", timeout=timeout, verify=verify, cafile=cafile)
        self._config = config
        self._request_id = itertools.count(1)
        self._session = Session()

    @property
    def url(self) -> str:
        return self._config.url

    @property
    def timeout(self) -> float:
        return self._config.timeout

    def call(self, method: str, params: Optional[Iterable[Any]] = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._request_id),
            "method": method,
            "params": list(params or []),
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            response = self._session.post(
                self.url,
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                verify=self._verify_param,
            )
            response.raise_for_status()
        except RequestException as exc:  # pragma: no cover - pass-through for visibility
            raise RuntimeError(f"Failed to reach Sui RPC endpoint: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive parsing guard
            raise RuntimeError("Invalid JSON response from Sui RPC endpoint") from exc
        if "error" in data:
            raise SuiRPCError(method, data["error"])
        return data.get("result")

    # Convenience wrappers -------------------------------------------------

    def get_transaction_block(self, digest: str, *, show_effects: bool = True, show_events: bool = False) -> Any:
        options = {
            "showInput": False,
            "showRawInput": False,
            "showEffects": show_effects,
            "showEvents": show_events,
            "showObjectChanges": False,
            "showBalanceChanges": False,
        }
        return self.call("sui_getTransactionBlock", [digest, options])

    _DEFAULT_OBJECT_OPTIONS = {
        "showContent": True,
        "showType": True,
        "showOwner": True,
        "showPreviousTransaction": True,
        "showStorageRebate": True,
    }

    def try_get_past_object(
        self,
        object_id: str,
        version: int,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        resolved = options or self._DEFAULT_OBJECT_OPTIONS
        return self.call("sui_tryGetPastObject", [object_id, version, resolved])

    def get_object(self, object_id: str, *, options: Optional[Dict[str, Any]] = None) -> Any:
        resolved = options or self._DEFAULT_OBJECT_OPTIONS
        return self.call("sui_getObject", [object_id, resolved])

    def get_dynamic_fields(
        self,
        parent_object_id: str,
        cursor: Optional[str] = None,
        limit: int = 100,
    ) -> Any:
        params: list[Any] = [parent_object_id, cursor, limit]
        return self._call_with_fallback(
            ["sui_getDynamicFields", "suix_getDynamicFields"], params
        )

    def get_dynamic_field_object(self, parent_object_id: str, name: Dict[str, Any]) -> Any:
        return self._call_with_fallback(
            ["sui_getDynamicFieldObject", "suix_getDynamicFieldObject"],
            [parent_object_id, name],
        )

    def _call_with_fallback(self, methods: Sequence[str], params: list[Any]) -> Any:
        last_error: Optional[SuiRPCError] = None
        for method in methods:
            try:
                return self.call(method, params)
            except SuiRPCError as exc:
                last_error = exc
                error_payload = exc.error
                if isinstance(error_payload, dict) and error_payload.get("code") == -32601:
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No RPC methods supplied for fallback call")

    @property
    def _verify_param(self) -> bool | str:
        if not self._config.verify:
            return False
        if self._config.cafile:
            return self._config.cafile
        return True


class WaterfallSuiRPCClient:
    """Wrap multiple RPC endpoints and try them sequentially on failure."""

    def __init__(
        self,
        urls: Sequence[str] | None = None,
        *,
        clients: Sequence[SuiRPCClient] | None = None,
        timeout: float = 30.0,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if clients is not None and urls is not None:
            raise ValueError("Provide either urls or clients, not both")
        if clients is not None:
            if not clients:
                raise ValueError("At least one client must be supplied")
            self._clients = list(clients)
        else:
            if not urls:
                raise ValueError("At least one RPC URL must be provided")
            base_kwargs: Dict[str, Any] = {"timeout": timeout}
            if client_kwargs:
                base_kwargs.update(client_kwargs)
            self._clients = [SuiRPCClient(url=url, **base_kwargs) for url in urls]

        self._last_success_index: int = 0

    def _iter_indices(self) -> Iterable[int]:
        count = len(self._clients)
        start = self._last_success_index % count
        yield start
        for offset in range(1, count):
            yield (start + offset) % count

    def _execute(self, operation: Callable[[SuiRPCClient], Any]) -> Any:
        errors: list[str] = []
        last_exc: Exception | None = None

        for index in self._iter_indices():
            client = self._clients[index]
            try:
                result = operation(client)
            except SuiRPCError as exc:
                errors.append(f"{client.url}: {exc.error}")
                last_exc = exc
                continue
            except RuntimeError as exc:
                errors.append(f"{client.url}: {exc}")
                last_exc = exc
                continue
            self._last_success_index = index
            return result

        error_message = "All RPC endpoints failed"
        if errors:
            error_message = f"{error_message}: {'; '.join(errors)}"
        if last_exc is not None:
            if isinstance(last_exc, SuiRPCError):
                raise SuiRPCError("waterfall", {"message": error_message}) from last_exc
            raise RuntimeError(error_message) from last_exc
        raise RuntimeError(error_message)

    def call(self, method: str, params: Optional[Iterable[Any]] = None) -> Any:
        return self._execute(lambda client: client.call(method, params))

    def get_transaction_block(self, digest: str, *, show_effects: bool = True, show_events: bool = False) -> Any:
        return self._execute(
            lambda client: client.get_transaction_block(
                digest,
                show_effects=show_effects,
                show_events=show_events,
            )
        )

    def try_get_past_object(
        self,
        object_id: str,
        version: int,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        errors: list[str] = []
        last_exc: RuntimeError | None = None

        for index in self._iter_indices():
            client = self._clients[index]
            try:
                result = client.try_get_past_object(object_id, version, options=options)
            except SuiRPCError as exc:
                errors.append(f"{client.url}: {exc.error}")
                last_exc = exc
                continue
            except RuntimeError as exc:
                errors.append(f"{client.url}: {exc}")
                last_exc = exc
                continue

            status = None
            if isinstance(result, dict):
                status = result.get("status")
                if status == "VersionFound":
                    self._last_success_index = index
                    return result

            errors.append(f"{client.url}: status={status or 'unknown'}")

        message = "All RPC endpoints failed to provide object version"
        if errors:
            message = f"{message}: {'; '.join(errors)}"
        if last_exc is not None:
            if isinstance(last_exc, SuiRPCError):
                raise SuiRPCError("waterfall_try_get_past_object", {"message": message}) from last_exc
            raise RuntimeError(message) from last_exc
        raise RuntimeError(message)

    def get_object(self, object_id: str, *, options: Optional[Dict[str, Any]] = None) -> Any:
        return self._execute(lambda client: client.get_object(object_id, options=options))

    def get_dynamic_fields(self, parent_object_id: str, cursor: Optional[str] = None, limit: int = 100) -> Any:
        return self._execute(
            lambda client: client.get_dynamic_fields(parent_object_id, cursor=cursor, limit=limit)
        )

    def get_dynamic_field_object(self, parent_object_id: str, name: Dict[str, Any]) -> Any:
        return self._execute(
            lambda client: client.get_dynamic_field_object(parent_object_id, name)
        )
