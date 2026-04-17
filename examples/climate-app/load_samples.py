#!/usr/bin/env python3
"""Load sample climate report packets and run the full verification pipeline.

Usage:
    python examples/climate-app/load_samples.py
    python examples/climate-app/load_samples.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

SAMPLES_DIR = Path(__file__).parent / "sample_packets"
POLL_INTERVAL = 2  # seconds between status polls
TIMEOUT = 300  # seconds max per pipeline run


def upload_packet(client: httpx.Client, base_url: str, packet_dir: Path) -> dict:
    """Upload a packet directory to POST /api/packets and return the response JSON."""
    meta = json.loads((packet_dir / "metadata.json").read_text())
    print(f"  Uploading: {meta['name']}")
    files = []
    report_path = packet_dir / meta["report_file"]
    files.append(("report", (report_path.name, report_path.read_bytes(), "text/plain")))
    for src_file in meta["source_files"]:
        src_path = packet_dir / src_file
        files.append(("sources", (src_path.name, src_path.read_bytes(), "text/plain")))
    resp = client.post(f"{base_url}/api/packets/", files=files)
    resp.raise_for_status()
    return resp.json()


def trigger_pipeline(client: httpx.Client, base_url: str, packet_id: str) -> str:
    """Trigger a full pipeline run via POST /api/packets/{packet_id}/run."""
    resp = client.post(f"{base_url}/api/packets/{packet_id}/run", json={})
    resp.raise_for_status()
    return resp.json()["run_version_id"]


def poll_until_done(client: httpx.Client, base_url: str, run_id: str) -> str:
    """Poll GET /api/runs/{run_id} every POLL_INTERVAL seconds until terminal state."""
    elapsed = 0
    while elapsed < TIMEOUT:
        resp = client.get(f"{base_url}/api/runs/{run_id}")
        resp.raise_for_status()
        status = resp.json().get("status", "unknown")
        if status in ("completed", "failed"):
            return status
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load sample packets into EvidenceEngine")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running EvidenceEngine API (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    if not SAMPLES_DIR.exists():
        print(f"ERROR: sample_packets directory not found at {SAMPLES_DIR}")
        sys.exit(1)

    packet_dirs = sorted(d for d in SAMPLES_DIR.iterdir() if d.is_dir())
    if not packet_dirs:
        print(f"ERROR: no packet directories found in {SAMPLES_DIR}")
        sys.exit(1)

    all_ok = True
    with httpx.Client(timeout=30) as client:
        for packet_dir in packet_dirs:
            print(f"\n[{packet_dir.name}]")
            try:
                packet = upload_packet(client, args.base_url, packet_dir)
                # PacketResponse returns 'id' (not 'packet_id')
                packet_id = packet["id"]
                print(f"  packet_id:     {packet_id}")

                run_id = trigger_pipeline(client, args.base_url, packet_id)
                print(f"  run_version_id: {run_id}")

                print("  Waiting for pipeline...", end="", flush=True)
                status = poll_until_done(client, args.base_url, run_id)
                print(f" {status.upper()}")

                if status != "completed":
                    print(f"  ERROR: pipeline did not complete (status={status})")
                    all_ok = False

            except httpx.HTTPStatusError as exc:
                print(f"\n  HTTP error: {exc.response.status_code} — {exc.response.text}")
                all_ok = False
            except httpx.RequestError as exc:
                print(f"\n  Connection error: {exc}")
                print("  Is the server running? Try: uv run uvicorn evidenceengine.api.app:app --reload")
                all_ok = False

    if all_ok:
        print("\nDone. Both packets completed successfully.")
        print("Open http://localhost:8000/dashboard to view results in the browser.")
    else:
        print("\nSome packets failed — check server logs for details.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
