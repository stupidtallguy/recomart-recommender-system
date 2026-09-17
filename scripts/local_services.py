"""Start/stop only this workspace's prepared Windows development services."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from time import monotonic, sleep

import redis
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
PG_BIN = Path(os.environ.get("RECOMART_POSTGRES_BIN", "C:/Program Files/PostgreSQL/18/bin"))


def configuration():
    cfg = json.loads((LOCAL / "services.json").read_text(encoding="utf-8"))
    for name, port in (("database_url", 55432), ("redis_url", 56379)):
        url = urlparse(cfg[name])
        if url.hostname != "127.0.0.1" or url.port != port:
            raise ValueError("This helper manages only the isolated local development endpoints")
    return cfg


def native(args, *, check=True):
    # Daemons can inherit pipe handles on Windows, keeping communicate() open
    # after pg_ctl or WSL exits. A log file lets us wait only for the launcher.
    with (LOCAL / "service_control.log").open("ab") as log:
        return subprocess.run(args, check=check, stdout=log, stderr=log, timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def postgres(action):
    configuration()
    cluster = (LOCAL / "postgres").resolve()
    if not cluster.is_relative_to(LOCAL.resolve()) or not (cluster / "PG_VERSION").exists():
        raise ValueError("Prepared project PostgreSQL cluster not found")
    ctl = str(PG_BIN / "pg_ctl.exe")
    running = native([ctl, "-D", str(cluster), "status"], check=False).returncode == 0
    if action == "start" and not running:
        native([ctl, "-D", str(cluster), "-l", str(LOCAL / "postgres.log"),
                "-o", "-h 127.0.0.1 -p 55432", "-w", "start"])
    elif action == "stop" and running:
        native([ctl, "-D", str(cluster), "-m", "fast", "-w", "stop"])


def redis_server(action):
    cfg = configuration()
    client = redis.Redis.from_url(cfg["redis_url"], socket_timeout=1, socket_connect_timeout=1)
    try:
        try:
            running = client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            running = False
        if action == "start" and not running:
            native(["wsl", "-d", "RecoMartRedis", "-u", "root", "--", "su", "-s", "/bin/sh", "redis",
                    "-c", "redis-server /etc/recomart-redis.conf"])
            deadline = monotonic() + 10
            while True:
                try:
                    client.ping()
                    break
                except (redis.ConnectionError, redis.TimeoutError):
                    if monotonic() >= deadline:
                        raise
                    sleep(.2)
        elif action == "stop" and running:
            client.shutdown(nosave=True)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop"])
    args = parser.parse_args()
    postgres(args.action)
    redis_server(args.action)
    print("Project PostgreSQL and Redis:", args.action)


if __name__ == "__main__":
    main()
