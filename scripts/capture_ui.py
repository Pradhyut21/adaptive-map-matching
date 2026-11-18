"""Capture actual local UI inference and the server's read-only log view.

This script never substitutes API responses or generates terminal images.
The images establish local integration only; a two-device demonstration is separate.
Install the optional capture dependency with `python -m pip install playwright`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trip", help="Discover the first available trip when omitted")
    parser.add_argument("--logs-path", default="/diagnostics/logs")
    parser.add_argument("--output", type=Path, default=Path("results/screenshots"))
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    args.output.mkdir(parents=True, exist_ok=True)
    metadata: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "ui_url": args.ui_url,
        "api_url": args.api_url,
        "evidence_scope": "Actual same-machine HTTP integration; second-laptop LAN demonstration remains pending.",
        "screenshots": [],
    }
    page_errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
        page.set_default_timeout(args.timeout_ms)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(args.ui_url, wait_until="domcontentloaded")
        page.locator("#connectButton").wait_for(state="visible")
        page.wait_for_function("!document.getElementById('connectButton').disabled")
        if page.locator("#serverUrl").input_value().rstrip("/") != args.api_url.rstrip("/"):
            page.locator("#serverUrl").fill(args.api_url)
            page.locator("#connectButton").click()
        page.wait_for_function("!document.getElementById('runButton').disabled")
        if args.trip:
            page.locator("#tripSelect").select_option(args.trip)
        metadata["trip_id"] = page.locator("#tripSelect").input_value()
        if not metadata["trip_id"]:
            raise RuntimeError("No real trip was discovered by the UI")
        with page.expect_response(
            lambda response: response.request.method == "POST" and response.url.rstrip("/").endswith("/predict"),
            timeout=args.timeout_ms,
        ) as received:
            page.locator("#runButton").click()
        response = received.value
        metadata["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError(f"Actual prediction request failed: {response.status} {response.text()[:500]}")
        result = response.json()
        page.wait_for_function("document.getElementById('runState').textContent === 'Complete'")
        page.wait_for_timeout(400)
        metadata["run_id"] = result.get("run_id")
        metadata["model_name"] = result.get("model_name")
        metadata["metrics"] = result.get("metrics")
        metadata["timing"] = result.get("timing")
        metadata["observation_count"] = len(result.get("raw_gps", []))
        metadata["warnings_count"] = len(result.get("warnings", []))
        metadata["desktop_horizontal_overflow"] = page.evaluate("document.body.scrollWidth > innerWidth")
        desktop = args.output / "ui_desktop.png"
        page.screenshot(path=str(desktop), full_page=True)
        metadata["screenshots"].append(str(desktop))

        # Exercise real display controls while retaining the same actual inference result.
        page.locator("#layerRoads").uncheck()
        page.locator("#layerRoads").check()
        page.locator("#zoomIn").click()
        page.locator("#fitButton").click()
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(250)
        page.locator("#fitButton").click()
        metadata["mobile_horizontal_overflow"] = page.evaluate("document.body.scrollWidth > innerWidth")
        mobile = args.output / "ui_mobile.png"
        page.screenshot(path=str(mobile), full_page=True)
        metadata["screenshots"].append(str(mobile))

        log_page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        log_response = log_page.goto(f"{args.api_url.rstrip('/')}{args.logs_path}", wait_until="domcontentloaded")
        metadata["logs_status"] = log_response.status if log_response else None
        if log_response and log_response.ok:
            log_text = log_page.locator("body").inner_text()
            if metadata["trip_id"] not in log_text:
                metadata["logs_note"] = "The read-only log page loaded; this view did not include the selected trip ID."
            logs = args.output / "api_logs.png"
            log_page.screenshot(path=str(logs), full_page=True)
            metadata["screenshots"].append(str(logs))
            metadata["log_image_scope"] = "Screenshot of actual server log text in the browser; not a terminal screenshot."
        else:
            metadata["logs_note"] = "Read-only log route unavailable; no log image fabricated."
        metadata["page_errors"] = page_errors
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        browser.close()
    destination = args.output / "capture_metadata.json"
    destination.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if page_errors or metadata["desktop_horizontal_overflow"] or metadata["mobile_horizontal_overflow"]:
        raise RuntimeError("Browser verification found errors; inspect capture_metadata.json")


if __name__ == "__main__":
    main()
