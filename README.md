# Multimodal Predictive Maintenance System

A full-stack industrial monitoring dashboard that combines **Graph Neural Networks (GNN)** and **NLP embeddings** for advanced failure prediction in industrial machinery.

## 🆕 Version 3.0 Features

- **JWT Authentication** - Secure email/password registration and login
- **Multi-tenant Support** - Users see only their own machines and data
- **PyTorch Geometric GNN** - Real GCN and GAT models trained on CMAPSS data
- **Real-time WebSocket Streaming** - Live sensor updates every 5 seconds
- **Alert System** - Automatic alerts with configurable thresholds
- **SendGrid Email Integration** - Optional email notifications for critical alerts

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Solution Architecture](#-solution-architecture)
3. [Features](#-features)
4. [Tech Stack](#-tech-stack)
5. [Authentication](#-authentication)
6. [GNN Models](#-gnn-models)
7. [Alert System](#-alert-system)
8. [API Reference](#-api-reference)
9. [Getting Started](#-getting-started)
10. [Data Formats](#-data-formats)

---

## 🎯 Problem Statement

Industrial machines (turbines, motors, compressors, CNC machines) generate massive sensor data. Traditional predictive maintenance approaches have limitations:

| Traditional Approach | Limitation |
|---------------------|------------|
| LSTM / RNN | Ignores sensor relationships |
| CNN anomaly detection | No relational modeling |
| Random Forest / XGBoost | Treats data as flat tables |
| Rule-based systems | Misses text data insights |

**Consequences of failure:**
- Unplanned downtime
- Production loss
- Expensive repairs
- Safety risks

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                          │
├─────────────────────────────────────────────────────────────────┤
│  Auth Page  │  Dashboard  │  Sensors  │  Prediction  │  Alerts  │
└──────┬──────┴──────┬──────┴─────┬─────┴──────┬───────┴────┬─────┘
       │             │            │            │            │
       │         WebSocket    REST API    REST API    REST API
       │             │            │            │            │
┌──────▼─────────────▼────────────▼────────────▼────────────▼─────┐
│                        BACKEND (FastAPI)                         │
├─────────────────────────────────────────────────────────────────┤
│  JWT Auth  │  GNN Engine  │  NLP Engine  │  Alert Engine        │
│            │  (PyTorch    │  (Keyword    │  (Threshold          │
│            │   Geometric) │   Extraction)│   + Email)           │
└──────┬─────┴──────┬───────┴──────┬───────┴──────┬───────────────┘
       │            │              │              │
       ▼            ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MongoDB                                   │
│  users │ machines │ sensor_readings │ predictions │ alerts      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
```
Sensor Data + Maintenance Logs
            ↓
     Graph Construction (Correlation Matrix)
            ↓
    Graph Neural Network (GCN + GAT ensemble)
            ↓
      NLP Embeddings (Risk keyword extraction)
            ↓
     Multimodal Fusion (Weighted combination)
            ↓
     Failure Prediction (RUL + Confidence)
            ↓
     Alert Engine (Threshold check + Email)
```

---

## ✨ Features

### 1. Authentication & Multi-tenancy
- **JWT-based auth** with bcrypt password hashing
- **User isolation** - each user sees only their own machines
- **Session persistence** with localStorage
- **Protected API routes** with Bearer token

### 2. Equipment Health Dashboard
- Real-time **health score gauge** (0-100)
- **Failure probability** percentage
- **Risk level badges** (Healthy/Warning/Critical)
- **Live sensor readings** with status indicators
- **Machine grid** for quick overview

### 3. Sensor Time-Series Viewer
- **Interactive Recharts** with area gradients
- **Sensor selector** (Temperature, Pressure, Vibration, RPM)
- **Time range slider** for data exploration
- **Statistics cards** (Min/Avg/Max)
- **Show All mode** to compare sensors

### 4. Failure Prediction Panel
- **Remaining Useful Life (RUL)** in days
- **Predicted failure date**
- **Confidence score** from ensemble model
- **Failure type** classification
- **Score breakdown**: GCN, GAT, NLP, Fusion

### 5. Sensor Dependency Graph
- **Force-directed visualization** with react-force-graph-2d
- **Sensor nodes** colored by type
- **Correlation edges** with weight-based thickness
- **Interactive** - click, zoom, pan

### 6. Maintenance Log Insights
- **NLP keyword extraction** from log text
- **Risk score** per log entry
- **Keyword cloud** of frequent issues
- **Severity filtering** and search

### 7. Alert Center
- **Real-time alerts** via WebSocket
- **Configurable thresholds** (Critical < 40%, Warning < 70%)
- **Email notifications** via SendGrid
- **Acknowledge workflow**

---

## 📦 Tech Stack

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.115+ | REST API framework |
| MongoDB + Motor | 3.11+ | Async database |
| PyTorch | 2.10+ | Deep learning |
| PyTorch Geometric | 2.7+ | GNN models (GCN, GAT) |
| bcrypt | 4.1+ | Password hashing |
| PyJWT | 2.11+ | JWT authentication |
| scipy | 1.14+ | Statistical computations |
| SendGrid | 6.12+ | Email notifications |

### Frontend
| Library | Version | Purpose |
|---------|---------|---------|
| React | 19+ | UI framework |
| Tailwind CSS | 3+ | Styling |
| Recharts | 2.13+ | Time-series charts |
| react-force-graph-2d | 1.26+ | Graph visualization |
| Framer Motion | 12+ | Animations |
| socket.io-client | 4.8+ | WebSocket |
| Lucide React | 0.469+ | Icons |
| shadcn/ui | - | UI components |

---

## 🔐 Authentication

### Registration
```bash
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "John Doe"
}
```

### Login
```bash
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Response
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "name": "John Doe",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### Using Protected Routes
```bash
curl -H "Authorization: Bearer <token>" https://api/machines
```

---

## 🧠 GNN Models

### Architecture

#### GCN (Graph Convolutional Network)
```python
class SensorGCN(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(4, 32)    # Input: 4 sensor features
        self.conv2 = GCNConv(32, 32)   # Hidden layer
        self.conv3 = GCNConv(32, 32)   # Hidden layer
        self.lin = nn.Linear(32, 3)    # Output: 3 classes
```

#### GAT (Graph Attention Network)
```python
class SensorGAT(nn.Module):
    def __init__(self):
        self.conv1 = GATConv(4, 32, heads=4)  # Multi-head attention
        self.conv2 = GATConv(128, 32, heads=1)
        self.lin = nn.Linear(32, 3)
```

### Graph Construction
1. **Nodes** = Sensors (temperature, pressure, vibration, rpm)
2. **Node Features** = [mean, std, min, max] of recent readings
3. **Edges** = Pearson correlations > 0.2 between sensors
4. **Edge Weights** = Correlation strength

### Prediction Classes
| Class | Label | RUL Range |
|-------|-------|-----------|
| 0 | Healthy | > 60 days |
| 1 | Warning | 30-60 days |
| 2 | Critical | < 30 days |

### Ensemble Method
```python
ensemble_probs = (gcn_output + gat_output) / 2
risk_score = ensemble[1] * 0.3 + ensemble[2] * 0.7
```

---

## 🔔 Alert System

### Threshold Configuration
| Health Score | Risk Level | Action |
|--------------|------------|--------|
| > 70% | Healthy | No alert |
| 40-70% | Warning | Yellow alert |
| < 40% | Critical | Red alert + Email |

### Alert Settings API
```bash
PUT /api/alert-settings
Authorization: Bearer <token>
Content-Type: application/json

{
  "email_enabled": true,
  "email_recipients": ["engineer@company.com"],
  "critical_threshold": 40,
  "warning_threshold": 70
}
```

### Email Notifications (Optional)
Add to `/app/backend/.env`:
```bash
SENDGRID_API_KEY=SG.your_key_here
SENDER_EMAIL=alerts@yourdomain.com
```

---

## 📡 API Reference

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Login |
| GET | `/api/auth/me` | Get current user |

### Machines
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/machines` | List user's machines |
| POST | `/api/machines` | Create machine |
| GET | `/api/machines/{id}` | Get machine details |
| DELETE | `/api/machines/{id}` | Delete machine |

### Sensor Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/machines/{id}/readings` | Get sensor readings |
| POST | `/api/machines/{id}/simulate` | Generate demo data |
| GET | `/api/machines/{id}/sensor-graph` | Get correlation graph |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/machines/{id}/predict` | Run GNN prediction |
| GET | `/api/machines/{id}/predictions` | Get history |

### Maintenance Logs
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/maintenance-logs` | Create log |
| GET | `/api/machines/{id}/maintenance-logs` | Get logs |

### Alerts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | Get alerts |
| POST | `/api/alerts/{id}/acknowledge` | Acknowledge |
| GET | `/api/alert-settings` | Get settings |
| PUT | `/api/alert-settings` | Update settings |

### Utilities
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/seed-demo` | Load demo data |
| POST | `/api/upload` | Upload CSV/JSON |
| GET | `/api/dashboard/summary` | Get stats |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `ws://host/ws/{machine_id}` | Real-time sensor stream |

---

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

### Quick Start
1. Open the application
2. Create an account (Sign Up)
3. Click "Load Demo Data"
4. Explore the dashboard
5. Click "Run Prediction" for GNN analysis

---

## 📊 Data Formats

### CSV Upload
```csv
timestamp,machine_id,temperature,pressure,vibration,rpm
2024-01-01T00:00:00,machine-1,45.2,102.5,0.52,3050
2024-01-01T01:00:00,machine-1,46.1,101.8,0.55,3020
```

### JSON Upload
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

---

## 🗂️ Project Structure

```
/app/
├── backend/
│   ├── server.py           # Main FastAPI application
│   ├── gnn_training.py     # GNN training script
│   ├── requirements.txt    # Python dependencies
│   ├── .env               # Environment variables
│   ├── data/              # CMAPSS training data
│   │   └── train_FD001.csv
│   └── models/            # Trained model weights
│       ├── gcn_cmapss.pt
│       └── gat_cmapss.pt
│
├── frontend/
│   ├── src/
│   │   ├── App.js                    # Main application
│   │   ├── components/
│   │   │   ├── AuthPage.jsx          # Login/Register
│   │   │   ├── Sidebar.jsx           # Navigation
│   │   │   ├── HealthDashboard.jsx   # Overview
│   │   │   ├── SensorTimeSeries.jsx  # Charts
│   │   │   ├── FailurePrediction.jsx # Predictions
│   │   │   ├── GraphVisualization.jsx # GNN graph
│   │   │   ├── MaintenanceLogs.jsx   # NLP logs
│   │   │   └── AlertsPanel.jsx       # Alerts
│   │   └── hooks/
│   │       └── useWebSocket.js       # Real-time hook
│   ├── package.json
│   └── .env
│
└── README.md
```

---

## 🔮 Roadmap

### Completed ✅
- [x] JWT authentication
- [x] Multi-tenant data isolation
- [x] PyTorch Geometric GCN/GAT
- [x] Real-time WebSocket streaming
- [x] Alert notification system
- [x] SendGrid email integration

### Coming Soon
- [ ] Train GNN on full NASA CMAPSS dataset
- [ ] Historical prediction accuracy tracking
- [ ] Export reports to PDF
- [ ] Role-based access control (Admin/Operator/Viewer)
- [ ] SMS alerts via Twilio
- [ ] Integration with industrial IoT platforms

---

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

## 🙏 Acknowledgments

- NASA Turbofan (CMAPSS) dataset for inspiration
- PyTorch Geometric team for GNN implementations
- Hugging Face for NLP research

---

Built with ❤️ using Emergent AI
