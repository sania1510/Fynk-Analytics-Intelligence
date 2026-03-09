#  Fynk — Finance Analytics Intelligence

> AI-powered financial analytics dashboard. Upload your financial data, get instant KPI insights, trend analysis, forecasts, anomaly detection, and natural language AI insights — all in one sleek interface.

---

## ✨ Features

###  Financial Analytics
- **KPI Dashboard** — Real-time revenue, profit, expenses, and custom metric tracking with period-over-period comparisons
- **Time Series Analysis** — Historical trend analysis across daily, weekly, monthly, and quarterly granularity
- **Dimension Breakdown** — Segment performance by product, region, category, or any dimension in your data
- **Forecasting Engine** — ML-powered predictions using Facebook Prophet or linear trend fallback
- **Anomaly Detection** — Z-score statistical outlier detection to catch unusual financial movements
- **Seasonality Analysis** — Day-of-week and monthly pattern detection with visual heatmaps

###  AI Intelligence
- **Gemini AI Insights** — Natural language insights that reference actual numbers from your data
- **Data Chat** — Ask questions about your financial data in plain English
- **Auto Schema Detection** — Automatically identifies metrics, dimensions, and time columns from any CSV/Excel file
- **Executive Summaries** — One-click board-ready summaries of your financial performance

###  Data Support
- CSV, Excel (.xlsx/.xls), and JSON file uploads
- Drag & drop interface
- Auto data cleaning and normalization
- Smart column type detection

---

## 🖥️ Screenshots

> Dashboard · Time Series · Forecast · Anomaly Detection · AI Chat

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip
- Google Gemini API key *(optional — AI insights work without it using rule-based engine)*

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fynk.git
cd fynk
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set environment variables
```bash
# Required for Gemini AI insights (optional but recommended)
export GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: specify Gemini model
export GEMINI_MODEL=gemini-2.5-flash
```

### 4. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend
Open `nexus.html` (or `fynk.html`) in your browser. That's it — no build step needed.

---

## 📁 Project Structure

```
fynk/
├── main.py                   # FastAPI app entry point
├── backend.py                # Alternative backend entry point
├── requirements.txt
├── fynk.html                 # Frontend (single-file UI)
│
└── src/
    ├── api/
    │   └── routes.py         # All API endpoints
    │
    ├── analytics/
    │   ├── analyzer.py       # Core analytics engine (KPI, time series, forecast, etc.)
    │   ├── insights.py       # Gemini AI + rule-based insights generator
    │   └── schema.py         # Pydantic data models
    │
    └── data/
        ├── loader.py         # File upload & dataset management
        ├── smart_detector.py # AI-powered schema auto-detection
        ├── normalizer.py     # Data cleaning & normalization
        └── column_matcher.py # Rule-based column classification
```

---

## 🔌 API Reference

### Data Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload CSV / Excel / JSON file |
| `GET` | `/api/datasets` | List all loaded datasets |
| `GET` | `/api/schema/{dataset_id}` | Get auto-detected schema |
| `GET` | `/api/preview/{dataset_id}` | Preview first N rows |
| `DELETE` | `/api/datasets/{dataset_id}` | Delete a dataset |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze/kpi` | KPI analysis with period comparison |
| `POST` | `/api/analyze/time-series` | Historical trend analysis |
| `POST` | `/api/analyze/breakdown` | Dimension breakdown |
| `POST` | `/api/analyze/forecast` | ML-powered forecasting |
| `POST` | `/api/analyze/anomalies` | Anomaly detection |
| `POST` | `/api/analyze/seasonality` | Seasonal pattern analysis |

### AI Insights
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/insights` | Generate AI insights from any analysis result |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/api/info` | Service status (ML, AI, etc.) |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.10+ |
| Data Processing | pandas, NumPy |
| ML Forecasting | Facebook Prophet *(optional)* |
| AI Insights | Google Gemini 2.5 Flash |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Orbitron, Rajdhani, Share Tech Mono |

---

##  Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | `None` | Gemini API key for AI insights |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model version |
| `PORT` | `8000` | Backend server port |

> **Note:** Fynk works fully without a Gemini API key. The rule-based insight engine generates real data-driven insights using the actual numbers from your dataset.

---

##  Sample Use Cases

- **Revenue tracking** — Upload monthly P&L data and instantly see revenue trends, growth rates, and forecasts
- **Expense analysis** — Break down expenses by category and identify the top cost drivers
- **Anomaly alerts** — Detect unusual spikes or drops in revenue, transactions, or any financial metric
- **Seasonal planning** — Understand which months and days perform best to plan campaigns and staffing
- **Board reporting** — Generate executive summaries in one click from your financial data

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request


---

## 🙋 Support

If you run into issues or have questions, open an issue on GitHub.

---

<p align="center">Built with ❤️ for finance teams who want smarter data</p>