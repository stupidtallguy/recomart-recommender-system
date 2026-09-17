"""Exercise outages only on this project's managed local development services.

Run while the API is listening on port 8000. Both backends are restored in a
finally block. Never use this against shared or production services.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import monotonic, sleep
from urllib.request import urlopen

from local_services import ROOT, configuration, postgres, redis_server


def get(path):
    with urlopen("http://127.0.0.1:8000" + path, timeout=10) as response:
        return json.load(response), {k.lower(): v for k, v in response.headers.items()}


def main():
    configuration()  # Rejects URLs outside the project-only loopback ports.
    original, _ = get("/recommendations/1?k=7")
    report = {"snapshot_id": original["snapshot_id"]}
    try:
        redis_server("stop")
        body, headers = get("/recommendations/1?k=7")
        assert body == original and headers["x-recomart-source"] == "postgresql"
        assert get("/health")[0]["cache"] == "unavailable"
        report["redis_outage"] = "same response from PostgreSQL"
        print("Redis outage fallback passed", flush=True)
        postgres("stop")
        body, headers = get("/recommendations/1?k=7")
        assert body == original and headers["x-recomart-source"] == "snapshot"
        health = get("/health")[0]
        assert health["ready"] and health["database"] == health["cache"] == "unavailable"
        report["both_backends_outage"] = "same response from sealed local snapshot"
        print("Combined outage fallback passed", flush=True)
    finally:
        try:
            postgres("start")
        finally:
            redis_server("start")
    deadline = monotonic() + 30
    while True:
        health = get("/health")[0]
        if health["database"] == health["cache"] == "ok":
            break
        if monotonic() >= deadline:
            raise RuntimeError("Backends started but API did not recover within 30 seconds")
        sleep(.5)
    body, _ = get("/recommendations/1?k=7")
    assert body == original
    body, headers = get("/recommendations/1?k=7")
    assert body == original and headers["x-recomart-source"] == "redis"
    report.update(status="passed", completed_at=datetime.now(timezone.utc).isoformat(),
        recovery="both backends healthy; Redis hits restored", health=health)
    path = ROOT / "results/benchmarks/backend_outage_smoke.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
