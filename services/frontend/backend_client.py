import json
import os
from typing import Any
from urllib.parse import quote
from urllib import error, request

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api/v1"

def normalize_backend_url(raw_url: str) -> str:
    normalized = (raw_url or "").strip().rstrip("/")
    if normalized and not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized

def _request_json(
    backend_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 12,
) -> tuple[int, dict[str, Any] | None, str | None]:
    url = f"{backend_url}{path}"
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=method)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body) if body else {}
            return response.status, parsed, None
    except error.HTTPError as http_error:
        raw_body = http_error.read().decode("utf-8")
        try:
            parsed = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw_body or "Unknown HTTP error"}
        return http_error.code, parsed, None
    except Exception as exc:
        return 0, None, str(exc)

def get_health(backend_url: str) -> dict[str, Any]:
    status_code, payload, transport_error = _request_json(
        backend_url=backend_url,
        method="GET",
        path=f"{API_PREFIX}/health",
    )
    if transport_error:
        return {"ok": False, "error": transport_error}
    if status_code == 200:
        return {"ok": True, "data": payload}
    if status_code == 503:
        detail = (payload or {}).get("detail", {})
        return {"ok": False, "data": detail, "error": "Service is unhealthy"}
    return {"ok": False, "error": (payload or {}).get("detail", f"HTTP {status_code}")}

def get_models(backend_url: str) -> dict[str, Any]:
    status_code, payload, transport_error = _request_json(
        backend_url=backend_url,
        method="GET",
        path=f"{API_PREFIX}/models",
    )
    if transport_error:
        return {"ok": False, "models": [], "error": transport_error}
    if status_code == 200:
        models = (payload or {}).get("models", [])
        return {"ok": True, "models": models}
    return {"ok": False, "models": [], "error": (payload or {}).get("detail", f"HTTP {status_code}")}

def predict_request(
    backend_url: str,
    text: str,
    model_name: str,
    model_alias: str = "production",
) -> dict[str, Any]:
    status_code, payload, transport_error = _request_json(
        backend_url=backend_url,
        method="POST",
        path=f"{API_PREFIX}/predict",
        payload={
            "text": text,
            "model_name": model_name,
            "model_alias": model_alias,
        },
    )
    if transport_error:
        return {"ok": False, "error": transport_error}
    if status_code == 200:
        return {
            "ok": True,
            "prediction": (payload or {}).get("prediction"),
            "prediction_label": (payload or {}).get("prediction_label"),
            "prediction_code": (payload or {}).get("prediction_code"),
            "model_name": (payload or {}).get("model_name"),
            "confidence": (payload or {}).get("confidence"),
        }
    return {"ok": False, "error": (payload or {}).get("detail", f"HTTP {status_code}")}

def get_model_info(backend_url: str, model_name: str, alias: str = "production") -> dict[str, Any]:
    safe_name = quote(model_name, safe="")
    status_code, payload, transport_error = _request_json(
        backend_url=backend_url,
        method="GET",
        path=f"{API_PREFIX}/model_info/{safe_name}?alias={quote(alias, safe='')}",
    )
    if transport_error:
        return {"ok": False, "error": transport_error, "versions": []}
    if status_code == 200:
        return {
            "ok": True,
            "model_name": (payload or {}).get("model_name", model_name),
            "versions": (payload or {}).get("versions", []),
        }
    return {
        "ok": False,
        "error": (payload or {}).get("detail", f"HTTP {status_code}"),
        "versions": [],
    }

def get_mlflow_models(backend_url: str) -> dict[str, Any]:
    status_code, payload, transport_error = _request_json(
        backend_url=backend_url,
        method="GET",
        path=f"{API_PREFIX}/mlflow_models",
    )
    if transport_error:
        return {"ok": False, "model_names": [], "error": transport_error}
    if status_code == 200:
        model_names = (payload or {}).get("model_names", [])
        return {"ok": True, "model_names": model_names}
    return {"ok": False, "model_names": [], "error": (payload or {}).get("detail", f"HTTP {status_code}")}
