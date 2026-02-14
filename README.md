# Multimodal Predictive Maintenance System

A full-stack industrial monitoring dashboard that combines **Graph Neural Networks (GNN)** and **NLP embeddings** for advanced failure prediction in industrial machinery.

## 🆕 Version 2.0 Features

- **PyTorch Geometric GNN** - Real GCN and GAT models for sensor dependency learning
- **Real-time WebSocket Streaming** - Live sensor updates every 5 seconds
- **Alert System** - Automatic alerts when health drops below thresholds
- **Email Notifications** - SendGrid integration for critical alerts (optional)

![Dashboard Preview](https://images.unsplash.com/photo-1701448149957-b96dbd1926ff?w=800)

## 🎯 Problem Statement

Industrial machines (turbines, motors, compressors, CNC machines) generate massive sensor data. Traditional predictive maintenance approaches have limitations:

- **Ignores sensor relationships** - Sensors don't work independently (vibration affects temperature, pressure affects RPM)
- **No relational modeling** - Traditional ML treats data as flat tables
- **Text data unused** - Maintenance logs contain valuable insights like "Abnormal bearing noise"
- **Poor contextual reasoning** - Models can't interpret semantic risk signals

## 💡 Solution

This system implements a **Multimodal Predictive Maintenance Framework** that combines:

1. **Graph Neural Networks (GNNs)** - Model sensor dependencies as a correlation graph
2. **NLP Embeddings** - Extract risk keywords from maintenance logs
3. **Time-series Analysis** - Track sensor degradation patterns
4. **Multimodal Fusion** - Combine all signals for enhanced prediction accuracy

```
Sensor Data + Maintenance Logs
            ↓
     Graph Construction
            ↓
    Graph Neural Network
            ↓
      NLP Embeddings
            ↓
     Multimodal Fusion
            ↓
     Failure Prediction
```

## 🏗️ Architecture

### Backend (FastAPI + MongoDB)

```
/app/backend/
├── server.py          # Main API server with 15+ endpoints
│   ├── Data Models    # Machine, SensorReading, MaintenanceLog, Prediction
│   ├── Simulation     # Realistic degradation pattern generator
│   ├── GNN Module     # Correlation graph construction & prediction
│   ├── NLP Module     # Risk keyword extraction & analysis
│   └── Fusion Engine  # Multimodal prediction combining GNN + NLP
├── requirements.txt   # Python dependencies
└── .env              # Environment configuration
```

### Frontend (React + Tailwind)

```
/app/frontend/src/
├── App.js                          # Main application with routing
├── components/
│   ├── Sidebar.jsx                 # Navigation & machine selector
│   ├── HealthDashboard.jsx         # Equipment health overview
│   ├── SensorTimeSeries.jsx        # Interactive sensor charts
│   ├── FailurePrediction.jsx       # RUL & prediction display
│   ├── GraphVisualization.jsx      # Force-directed sensor graph
│   └── MaintenanceLogs.jsx         # NLP-analyzed log entries
└── index.css                       # Dark industrial theme styles
```

## 🎨 UI Design

The interface follows an **"Obsidian Control Room"** aesthetic - a dark industrial theme inspired by factory control rooms:

| Element | Color |
|---------|-------|
| Background | Dark zinc (#09090b) |
| Healthy Status | Emerald (#10b981) |
| Warning Status | Yellow (#facc15) |
| Critical Status | Red (#ef4444) |
| Accent/Highlights | Cyan (#00f0ff) |

**Typography:**
- **Headings**: Chivo (technical, industrial feel)
- **Data/Numbers**: JetBrains Mono (monospace for readings)
- **Body**: Inter (clean readability)

## 📊 Features

### 1. Equipment Health Dashboard
- **Health Score Gauge** (0-100) with animated ring
- **Failure Probability** percentage with progress bar
- **Risk Level Badges** (Healthy/Warning/Critical)
- **Live Sensor Readings** with status indicators
- **Machine Grid** for quick overview of all equipment

### 2. Sensor Time-Series Viewer
- **Interactive Charts** using Recharts with area gradients
- **Sensor Selector** (Temperature, Pressure, Vibration, RPM)
- **Time Range Slider** for data exploration
- **Statistics Cards** showing Min/Avg/Max values
- **Show All Mode** to compare all sensors simultaneously

### 3. Failure Prediction Panel
- **Remaining Useful Life (RUL)** in days
- **Predicted Failure Date** with calendar display
- **Confidence Score** percentage
- **Failure Type** classification
- **Score Breakdown**: GNN Score, NLP Score, Fusion Score
- **Prediction History** for trend analysis

### 4. Sensor Dependency Graph
- **Force-Directed Visualization** using react-force-graph-2d
- **Sensor Nodes** colored by type (Temperature=Red, Pressure=Blue, etc.)
- **Correlation Edges** with weight-based thickness
- **Interactive** - click nodes for details, zoom/pan controls
- **Graph Statistics** - node count, edge count, average correlation

### 5. Maintenance Log Insights
- **NLP Analysis** with SentenceTransformers
- **Risk Keyword Extraction** (abnormal, noise, leak, bearing, etc.)
- **Risk Score** per log entry (0-100%)
- **Keyword Cloud** showing most frequent risk indicators
- **Severity Filtering** (Info/Warning/Error/Critical)
- **Search** across logs, technicians, and keywords

## 🔌 API Endpoints

### Machines
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/machines` | List all machines |
| POST | `/api/machines` | Create new machine |
| GET | `/api/machines/{id}` | Get machine details |
| DELETE | `/api/machines/{id}` | Delete machine |

### Sensor Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/machines/{id}/readings` | Get sensor readings |
| POST | `/api/machines/{id}/simulate` | Generate simulated data |
| GET | `/api/machines/{id}/sensor-graph` | Get correlation graph |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/machines/{id}/predict` | Run multimodal prediction |
| GET | `/api/machines/{id}/predictions` | Get prediction history |

### Maintenance Logs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/maintenance-logs` | Create log (with NLP analysis) |
| GET | `/api/machines/{id}/maintenance-logs` | Get logs for machine |

### Utilities
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/seed-demo` | Load demo data |
| POST | `/api/upload` | Upload CSV/JSON sensor data |
| GET | `/api/dashboard/summary` | Get overall statistics |

## 🧠 How the ML Pipeline Works

### 1. Data Simulation
```python
def generate_degradation_pattern(days):
    # Exponential degradation with noise
    degradation = 1 - np.exp(-3 * (x ** 2))
    noise = np.random.normal(0, 0.02, days)
    return degradation + noise
```

Simulates realistic run-to-failure trajectories with:
- Gradual sensor degradation
- Correlated sensor changes
- Random anomaly spikes (2% probability scaled by degradation)

### 2. Graph Construction
```python
def build_sensor_correlation_graph(readings):
    # Calculate Pearson correlations between sensors
    for each sensor pair:
        correlation = pearsonr(sensor1_data, sensor2_data)
        if abs(correlation) > 0.3:  # Significant correlation
            add_edge(sensor1, sensor2, weight=correlation)
```

Creates a graph where:
- **Nodes** = Sensors (temperature, pressure, vibration, rpm)
- **Edges** = Correlations between sensors
- **Edge Weights** = Correlation strength

### 3. GNN Prediction
```python
def gnn_predict(graph, readings):
    # Extract graph features
    total_edge_weight = sum(edge weights)
    node_variance = sum(node variances)
    
    # Sensor trend analysis
    temp_trend = recent_temp / initial_temp
    vib_trend = recent_vib / initial_vib
    
    # Combine for prediction score
    score = 0.3*edge_weight + 0.2*variance + 0.25*temp_trend + 0.25*vib_trend
```

### 4. NLP Analysis
```python
RISK_KEYWORDS = ["abnormal", "noise", "leak", "vibration", "overheating", 
                 "failure", "broken", "crack", "worn", "bearing", ...]

def analyze_maintenance_log(text):
    found_keywords = [kw for kw in RISK_KEYWORDS if kw in text.lower()]
    risk_score = min(1.0, len(found_keywords) * 0.15)
    return found_keywords, risk_score
```

### 5. Multimodal Fusion
```python
def multimodal_fusion_predict(gnn_score, nlp_score, health_score):
    # Weighted combination
    fusion_score = 0.4*gnn_score + 0.3*nlp_score + 0.3*(1 - health_score/100)
    
    # Map to RUL and failure type
    if fusion_score < 0.2: rul = 90 days, type = "None predicted"
    elif fusion_score < 0.4: rul = 60 days, type = "Minor wear"
    elif fusion_score < 0.6: rul = 30 days, type = "Component degradation"
    elif fusion_score < 0.8: rul = 14 days, type = "Bearing failure likely"
    else: rul = 7 days, type = "Imminent failure"
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB

### Backend Setup
```bash
cd /app/backend
pip install -r requirements.txt
# Server runs on http://localhost:8001
```

### Frontend Setup
```bash
cd /app/frontend
yarn install
yarn start
# App runs on http://localhost:3000
```

### Load Demo Data
1. Open the application
2. Click "Load Demo Data" in the sidebar
3. This creates 5 demo machines with:
   - 90 days of simulated sensor data
   - Maintenance logs with risk keywords
   - Various health states (healthy, warning, critical)

### Upload Custom Data
Supported formats: **CSV** and **JSON**

CSV format:
```csv
timestamp,machine_id,temperature,pressure,vibration,rpm
2024-01-01T00:00:00,machine-1,45.2,102.5,0.52,3050
```

JSON format:
```json
[
  {
    "machine_id": "machine-1",
    "timestamp": "2024-01-01T00:00:00",
    "temperature": 45.2,
    "pressure": 102.5,
    "vibration": 0.52,
    "rpm": 3050
  }
]
```

## 📦 Tech Stack

### Backend
| Library | Purpose |
|---------|---------|
| FastAPI | REST API framework |
| MongoDB + Motor | Async database |
| PyTorch | Deep learning framework |
| PyTorch Geometric | GNN models (GCN, GAT) |
| scipy | Statistical computations |
| networkx | Graph algorithms |
| scikit-learn | ML utilities |
| sentence-transformers | NLP embeddings |
| SendGrid | Email notifications |

### Frontend
| Library | Purpose |
|---------|---------|
| React 19 | UI framework |
| Tailwind CSS | Styling |
| Recharts | Time-series charts |
| react-force-graph-2d | Graph visualization |
| Framer Motion | Animations |
| socket.io-client | WebSocket connection |
| Lucide React | Icons |
| shadcn/ui | UI components |

## 🔔 Alert System

The system monitors machine health and triggers alerts based on configurable thresholds:

| Health Score | Risk Level | Action |
|--------------|------------|--------|
| > 70% | Healthy | No alert |
| 40-70% | Warning | Yellow alert |
| < 40% | Critical | Red alert + optional email |

### Configuring Alerts

1. Go to **Alerts** tab in sidebar
2. Click **Alert Settings**
3. Configure:
   - Email notifications toggle
   - Email recipients list
   - Critical threshold (default: 40%)
   - Warning threshold (default: 70%)

### Email Alerts (Optional)

To enable email alerts, add to `/app/backend/.env`:
```bash
SENDGRID_API_KEY=your_sendgrid_api_key
SENDER_EMAIL=alerts@yourdomain.com
```

## 🌐 WebSocket Real-time Streaming

The system streams live sensor data every 5 seconds when viewing a machine:

```
Live Sensors → WebSocket → Buffer
                          ↓
                  Graph Builder
                          ↓
                      GNN
                          ↓
                 Fusion + NLP
                          ↓
                 Health Score
                          ↓
        Dashboard + Alert Engine
```

WebSocket endpoint: `wss://your-domain/ws/{machine_id}`

## 🔮 Future Enhancements

### Completed in v2.0 ✅
- [x] PyTorch Geometric GCN/GAT models
- [x] Real-time WebSocket sensor streaming
- [x] Alert notification system
- [x] SendGrid email integration

### P2 - Medium Priority
- [ ] User authentication system
- [ ] Historical prediction accuracy tracking
- [ ] Export reports to PDF
- [ ] Mobile-responsive improvements
- [ ] Train GNN on real failure datasets

### P3 - Future
- [ ] Dynamic temporal graphs
- [ ] Comparison with baseline ML models (LSTM, Random Forest)
- [ ] Integration with industrial IoT platforms (OPC-UA, MQTT)
- [ ] SMS alerts via Twilio

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🙏 Acknowledgments

- NASA Turbofan (CMAPSS) dataset for inspiration
- PyTorch Geometric team for GNN implementations
- Hugging Face for SentenceTransformers

---

Built with ❤️ using Emergent AI
