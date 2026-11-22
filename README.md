# 🛰️ LIFELINE AI — Advanced Flood Detection & Prediction Command Center

[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat&logo=Streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=Python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero-Key](https://img.shields.io/badge/Engine-Zero--API--Key%20Autonomous-00f0ff.svg)](#)

> **Autonomous multi-horizon hydrological AI, satellite SAR inundation fusion, proactive disaster force mobilization, and real-time emergency alert broadcasting.**

---

## 🌟 Key Capabilities

### 1. 🛰️ Retrospective Flood Detection ("What Happened")
- **Dual-Engine Architecture**: Runs seamlessly on the **Autonomous Geospatial Engine** (Zero API keys required!) using ERA5 precipitation, GloFAS river surge physics, and OpenStreetMap hydrography, or optionally connects to **Google Earth Engine (Sentinel-1 SAR)**.
- **Inundation Mapping**: 100% free unauthenticated interactive GIS maps (OpenStreetMap, Esri World Imagery, OpenTopoMap).
- **Humanitarian Resource Requisition**: Calculates clean drinking water, food ration packs, and family shelters adhering strictly to **UN Sphere Standards**.
- **Critical Infrastructure Search**: Queries live OpenStreetMap Overpass API for nearby hospitals, clinics, schools, and candidate evacuation centers with direct Google Maps navigation routing.

### 2. 🔮 Predictive Flood Forecasting ("What Will Happen")
- **7-Horizon Parallel ML Engine**: Chronological stacking ensemble (`HistGradientBoosting` + `ExtraTrees` + `LogisticRegression`) paired with quantile regressors ($q=0.50, q=0.90$) trained on multi-year hydrological archives.
- **What-If Storm Stress-Testing**: Simulate instant storm surges (+25mm to +150mm) to observe real-time flood risk shifts without retraining.
- **Live Regional Telemetry**: Open-Meteo NWP forecasts, wind gusts, atmospheric pressure, marine wave surges, and UN GDACS disaster alerts.
- **Cyclone Tracking**: Automated IMD track rendering for active storms (e.g., Severe Cyclonic Storm Akvan).

### 3. 🛡️ Pre-Disaster Force Mobilization & Resource Requisition
- **Tactical Force Calculations**: Forecasts required **NDRF / SDRF Search & Rescue Teams**, commanding officers, police cordon personnel, motorized inflatable rescue boats (IRBs with 40HP OBM motors), and heavy 4x4 troop transports.
- **Staging Base Positioning**: Designates primary river-basin staging coordinates and sets advance pre-positioning deadlines ($T-24\text{h}$, $T-48\text{h}$, or $T-72\text{h}$ before flood peak).
- **Official Force Deployment Directive**: Generates and exports downloadable, formal military/SDMA command orders (`.txt`).

### 4. 📲 Multi-Recipient Emergency Alert & Broadcast Console
- **Multi-Audience Templates**: Public Early-Warning, NDRF/SDRF Tactical Mobilization, District Magistrate / Administrative Directives, and Gram Panchayat Notices (available in English and हिन्दी).
- **Multi-Phone Number Input**: Enter multiple contacts (comma, space, or newline separated) with real-time validation and normalization.
- **1-Click WhatsApp Direct Dispatch**: Instant click-to-send links for each recipient on web and mobile.
- **Native Device SMS & Simulated Broadcast**: One-tap SMS app launch and simulated national emergency gateway dispatch with delivery receipts.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Internet connection (for free public map tiles and weather telemetry)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/lifeline-ai-flood.git
   cd lifeline-ai-flood
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Command Center**:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Project Structure

```text
lifeline-ai-flood/
├── .streamlit/
│   └── config.toml          # Custom dark command-center theme
├── app.py                   # Main unified Streamlit application (3,200+ lines)
├── requirements.txt         # Required Python packages
├── .gitignore               # Excludes virtual environments and temporary caches
└── README.md                # Project documentation
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
