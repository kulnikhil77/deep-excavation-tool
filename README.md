# Deep Excavation Analysis Tool

Web-based sheet pile wall design tool built on Indian Standards (IS codes).

## Features

- **Anchored Wall** — Beam FE analysis with rebar, SDA, and prestressed anchors. Wind barrier loading per IS 875-3.
- **Cantilever Wall** — Free Earth Support and Blum methods with automatic embedment.
- **Staged Excavation** — Construction sequence simulation with auto stage generation and BM/SF/deflection envelope across all stages.
- **Section Library** — 110 sheet pile sections (ArcelorMittal, JFE, SAIL, Nippon) with IS 800:2007 capacity checks, auto-selection by demand.

## IS Code Compliance

- IS 9527:1980 — Sheet pile walls
- IS 14458 (Part 1):1998 — Retaining wall design
- IS 800:2007 — Steel structures (LSM)
- IS 456:2000 — Reinforced concrete (anchor design)
- IS 875 (Part 3):2015 — Wind loads
- IS 1893:2016 — Seismic loads (coming soon)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Deploy

## Tech Stack

Python, Streamlit, Plotly, NumPy — no external solvers required.
