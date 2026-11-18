# Two-laptop deployment

The inference pipeline runs on Laptop 1. Laptop 2 serves only the static UI and sends real HTTP requests to Laptop 1. The frontend has no dependency on external map tiles, CDNs, routing services or internet access. It draws the actual returned GPS observations and road geometry on a local canvas. The backend remains the authority for feature extraction, scoring, decoding and aggregate evaluation.

## Laptop 1: model and API

From the project root, install dependencies and set the permitted dataset location. The root README describes training and required artifacts.

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_DATA = 'D:\path\to\Student folder'
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's Wi-Fi or Ethernet IPv4 address using `ipconfig`, for example `192.168.1.10`. Both laptops must be on the same accessible LAN. If Windows prompts about Python's network access, allow the intended private network according to your institution's rules. This project does not alter firewall settings automatically.

Verify `http://127.0.0.1:8000/health` on Laptop 1. Leave its terminal visible during evaluation: actual `POST /predict` requests must appear in its logs.

## Laptop 2: evaluator UI

Copy the `frontend` folder or checkout the repository. Only Python's static server is needed on this laptop.

```powershell
python -m http.server 8800 --directory frontend --bind 0.0.0.0
```

Open `http://127.0.0.1:8800` on Laptop 2. Enter Laptop 1's LAN URL, for example `http://192.168.1.10:8000`, in **Inference server** and select **Connect**. The server address is saved locally in that browser. Choose a discovered trip and select **Reconstruct route**.

The UI calls `GET /health`, `GET /trips` and `POST /predict` with `{"trip_id":"selected-id"}`. The backend must allow the UI's actual origin through CORS. Avoid opening `index.html` as a `file://` page; serving it over HTTP gives a predictable origin.

For a same-laptop integration check, serve the UI on port 8800 and use API address `http://127.0.0.1:8000`. This is useful integration evidence but does not establish that the required second-laptop demonstration occurred.

## Evaluation and ground-truth isolation

- The UI discovers trip IDs from the API; IDs are not hardcoded.
- It renders raw GPS, road-network geometry and predicted route segments from real API responses.
- It displays server-computed aggregate metrics. Reference routes and per-observation reference labels must never be returned to the browser.
- Unsupported gaps appear as warnings and separate predicted geometries. The frontend does not connect separate route segments.
- The observation inspector exposes predicted arc IDs and optional observable diagnostics only. Score margin is an uncalibrated ambiguity indicator, not final-route confidence.
- The local display's equirectangular projection is for visualisation only. The pipeline's metric coordinate projection is implemented separately on the server.
- The mandatory Folium map is a separate artifact. This canvas UI does not replace it.

## Actual demonstration evidence

**Second-laptop LAN demonstration: PENDING until performed with the team's second device.**

During that demonstration, capture the actual Laptop 2 UI showing a completed trip and Laptop 1's terminal showing the receiving API requests. Record the time, trip, API address and response run ID. Keep screenshots with the experiment evidence and update `STATUS.md` only after the demonstration occurs. Do not substitute generated terminal images or a same-machine test for two-device evidence.
