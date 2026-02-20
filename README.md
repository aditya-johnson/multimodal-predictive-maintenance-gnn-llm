# 🔧 PredictMaint - Multimodal Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Multimodal Predictive Maintenance** platform that combines **Graph Neural Networks (GNN)**, **Natural Language Processing (NLP)**, and **Time-Series Analysis** to predict industrial machine failures before they occur.

![Dashboard Preview](https://via.placeholder.com/800x400?text=PredictMaint+Dashboard)

## 🎯 Key Results

| Metric | GNN Fusion | Threshold Baseline | Improvement |
|--------|-----------|-------------------|-------------|
| **F1 Score** | 89.7% | 79.1% | +10.6% |
| **Accuracy** | 89.4% | 77.1% | +12.3% |
| **Early Warning** | 11.9 days | ~0 days | +11.9 days |
| **False Positive Rate** | 6.1% | 12.2% | -50% |
| **Annual ROI** | 55% | - | $134,393 savings |

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Machine Learning Models](#-machine-learning-models)
- [Database Schema](#-database-schema)
- [Project Structure](#-project-structure)
- [Training the Model](#-training-the-model)
- [Contributing](#-contributing)

---

## ✨ Features

### 🏭 Equipment Health Monitoring
- **Real-time Dashboard** - Live health scores for all monitored machines
- **Multi-sensor Tracking** - Monitor 21+ sensor types (temperature, vibration, pressure, etc.)
- **Degradation Visualization** - Interactive charts showing equipment degradation over time
- **WebSocket Streaming** - Real-time updates every 5 seconds

### 🤖 AI-Powered Predictions
- **GNN Fusion Model** - Combines GCN + GAT architectures for superior accuracy
- **Remaining Useful Life (RUL)** - Predicts cycles until failure with 13.8 MAE
- **Health Classification** - Healthy / Warning / Critical states
- **Confidence Scores** - Prediction confidence levels for informed decisions

### 📊 Model Comparison Dashboard
- **Accuracy Metrics** - Side-by-side comparison with threshold baselines
- **Performance Radar** - Multi-dimensional model evaluation
- **Training History** - F1 Score and Loss curves over epochs
- **ROI Calculator** - Interactive cost savings estimator

### 👥 Enterprise Features
- **Role-Based Access Control (RBAC)** - Admin / Operator / Viewer roles
- **Multi-tenant Organizations** - Team-based data isolation
- **User Invitations** - Email-based team member invitations
- **PDF Report Generation** - Downloadable maintenance reports

### 🔔 Alert System
- **Configurable Thresholds** - Custom alert triggers per machine
- **Alert Acknowledgment** - Track and manage alert lifecycle
- **Critical Notifications** - Priority-based alert sorting

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend (React)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Health  │ │  Sensor  │ │ Failure  │ │  Graph   │ │Comparison│  │
│  │Dashboard │ │TimeSeries│ │Prediction│ │   Viz    │ │Dashboard │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend (FastAPI)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   Auth   │ │   RBAC   │ │Prediction│ │  Alerts  │ │  Reports │  │
│  │  (JWT)   │ │  Engine  │ │  Engine  │ │  System  │ │   (PDF)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌──────────┐  ┌──────────────┐  ┌──────────┐
             │ MongoDB  │  │ GNN Models   │  │WebSocket │
             │(Database)│  │(PyTorch Geo) │  │(Real-time)│
             └──────────┘  └──────────────┘  └──────────┘
```

### Data Flow

1. **Sensor Data Ingestion** → Raw readings from industrial equipment
2. **Graph Construction** → Build correlation-based sensor graphs
3. **GNN Processing** → Extract spatial-temporal features
4. **NLP Analysis** → Parse maintenance logs for risk keywords
5. **Fusion Prediction** → Combine GNN + NLP for final predictions
6. **Real-time Delivery** → WebSocket streaming to dashboard

---

## 🛠 Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance async API framework |
| **PyTorch** | Deep learning framework |
| **PyTorch Geometric** | Graph neural network library |
| **Motor** | Async MongoDB driver |
| **PyJWT** | JWT authentication |
| **ReportLab** | PDF generation |
| **WebSockets** | Real-time streaming |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 19** | UI framework |
| **Tailwind CSS** | Utility-first styling |
| **Recharts** | Data visualization |
| **react-force-graph-2d** | Graph visualization |
| **Framer Motion** | Animations |
| **Shadcn/UI** | Component library |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| **GCN (Graph Convolutional Network)** | Spatial feature extraction |
| **GAT (Graph Attention Network)** | Attention-based aggregation |
| **Ensemble Fusion** | Combined predictions |
| **SentenceTransformers** | NLP embeddings |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| **MongoDB** | Document database |
| **Supervisor** | Process management |
| **Hot Reload** | Development workflow |

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB 6.0+
- 4GB+ RAM (8GB recommended for training)

### Backend Setup

```bash
# Clone repository
git clone https://github.com/your-org/predictmaint.git
cd predictmaint

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB URL

# Start server
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
yarn install

# Configure environment
cp .env.example .env
# Set REACT_APP_BACKEND_URL

# Start development server
yarn start
```

### Environment Variables

**Backend (.env)**
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=predictmaint
JWT_SECRET=your-secret-key-here
```

**Frontend (.env)**
```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

---

## 📖 Usage

### 1. Register & Login

```bash
# Register a new user
curl -X POST http://localhost:8001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@company.com",
    "password": "securepassword",
    "fullname": "Admin User"
  }'

# Login
curl -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@company.com",
    "password": "securepassword"
  }'
```

### 2. Create Organization

```bash
curl -X POST http://localhost:8001/api/organizations \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Manufacturing Plant A"}'
```

### 3. Load Demo Data

```bash
curl -X POST http://localhost:8001/api/seed-demo \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Run Prediction

```bash
curl -X POST http://localhost:8001/api/machines/MACHINE_ID/predict \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 5. Download PDF Report

```bash
curl -X GET http://localhost:8001/api/machines/MACHINE_ID/report \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o report.pdf
```

---

## 📚 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login and get JWT token |
| GET | `/api/auth/me` | Get current user info |

### Organization Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/organizations` | Create organization |
| GET | `/api/organizations` | List user's organizations |
| POST | `/api/organizations/{id}/switch` | Switch active organization |
| POST | `/api/organizations/{id}/invite` | Invite member |
| GET | `/api/organizations/{id}/members` | List members |
| PUT | `/api/organizations/{id}/members/{user_id}/role` | Update member role |
| DELETE | `/api/organizations/{id}/members/{user_id}` | Remove member |

### Machine Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/machines` | List all machines |
| POST | `/api/machines` | Create machine |
| GET | `/api/machines/{id}` | Get machine details |
| POST | `/api/machines/{id}/predict` | Run prediction |
| GET | `/api/machines/{id}/report` | Download PDF report |
| GET | `/api/machines/{id}/readings` | Get sensor readings |

### Alert Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | List all alerts |
| POST | `/api/alerts/{id}/acknowledge` | Acknowledge alert |
| GET | `/api/alerts/settings` | Get alert settings |
| PUT | `/api/alerts/settings` | Update alert settings |

### Model Comparison

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/model-comparison` | Get GNN vs Threshold metrics |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://host/ws/{machine_id}` | Real-time sensor streaming |

---

## 🧠 Machine Learning Models

### Graph Neural Network Architecture

```
Input: Sensor Graph (14 nodes, correlation-based edges)
       Node Features: [mean, std, min, max, trend] per sensor

┌─────────────────────────────────────────────────────────────┐
│                     GNN Fusion Model                        │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │         GCN Branch      │  │        GAT Branch       │  │
│  │  ┌──────────────────┐   │  │  ┌──────────────────┐   │  │
│  │  │ GCNConv (5→64)   │   │  │  │ GATConv (5→16×4) │   │  │
│  │  │ BatchNorm + ReLU │   │  │  │ BatchNorm + ELU  │   │  │
│  │  ├──────────────────┤   │  │  ├──────────────────┤   │  │
│  │  │ GCNConv (64→64)  │   │  │  │ GATConv (64→16×4)│   │  │
│  │  │ BatchNorm + ReLU │   │  │  │ BatchNorm + ELU  │   │  │
│  │  ├──────────────────┤   │  │  ├──────────────────┤   │  │
│  │  │ GCNConv (64→64)  │   │  │  │ GATConv (64→64)  │   │  │
│  │  │ BatchNorm        │   │  │  │ BatchNorm        │   │  │
│  │  └──────────────────┘   │  │  └──────────────────┘   │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│              │                            │                 │
│              └──────────┬─────────────────┘                 │
│                         ▼                                   │
│              ┌─────────────────────┐                        │
│              │   Global Mean Pool  │                        │
│              └─────────────────────┘                        │
│                         │                                   │
│         ┌───────────────┴───────────────┐                   │
│         ▼                               ▼                   │
│  ┌─────────────────┐            ┌─────────────────┐         │
│  │  Classifier     │            │   Regressor     │         │
│  │  (64→32→3)      │            │   (64→32→1)     │         │
│  │  Health Class   │            │   RUL Pred      │         │
│  └─────────────────┘            └─────────────────┘         │
└─────────────────────────────────────────────────────────────┘

Output: Health Class (Healthy/Warning/Critical) + RUL (cycles)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | NASA CMAPSS FD001 |
| Engines | 100 turbofan units |
| Sensors | 14 (filtered from 21) |
| Window Size | 30 cycles |
| Batch Size | 32 |
| Epochs | 25 |
| Learning Rate | 0.001 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealing |
| Loss | CrossEntropy + MSE |

### Sensor Features Used

| Sensor | Description |
|--------|-------------|
| s2 | LPC outlet temperature |
| s3 | HPC outlet temperature |
| s4 | LPT outlet temperature |
| s7 | Total HPC outlet pressure |
| s8 | Physical fan speed |
| s9 | Physical core speed |
| s11 | Static HPC outlet pressure |
| s12 | Fuel flow ratio |
| s13 | Corrected fan speed |
| s14 | Corrected core speed |
| s15 | Bypass ratio |
| s17 | Bleed enthalpy |
| s20 | HPT coolant bleed |
| s21 | LPT coolant bleed |

---

## 🗄 Database Schema

### Users Collection
```javascript
{
  _id: ObjectId,
  email: String (unique),
  fullname: String,
  hashed_password: String,
  organization_id: ObjectId (nullable),
  role: String ("admin" | "operator" | "viewer"),
  created_at: DateTime
}
```

### Organizations Collection
```javascript
{
  _id: ObjectId,
  name: String,
  owner_id: ObjectId,
  created_at: DateTime
}
```

### Machines Collection
```javascript
{
  _id: ObjectId,
  name: String,
  type: String,
  organization_id: ObjectId,
  sensors: [{
    name: String,
    unit: String,
    min_threshold: Number,
    max_threshold: Number
  }],
  health_score: Number (0-100),
  status: String,
  created_at: DateTime
}
```

### Sensor Readings Collection
```javascript
{
  _id: ObjectId,
  machine_id: ObjectId,
  timestamp: DateTime,
  sensor_values: {
    temperature: Number,
    vibration: Number,
    pressure: Number,
    // ... other sensors
  }
}
```

### Predictions Collection
```javascript
{
  _id: ObjectId,
  machine_id: ObjectId,
  timestamp: DateTime,
  health_class: Number (0|1|2),
  health_label: String,
  rul_cycles: Number,
  confidence: Number,
  failure_probability: Number,
  model_version: String
}
```

### Alerts Collection
```javascript
{
  _id: ObjectId,
  machine_id: ObjectId,
  organization_id: ObjectId,
  type: String,
  severity: String ("low" | "medium" | "high" | "critical"),
  message: String,
  acknowledged: Boolean,
  acknowledged_by: ObjectId,
  created_at: DateTime
}
```

---

## 📁 Project Structure

```
/app
├── backend/
│   ├── server.py              # Main FastAPI application
│   ├── cmapss_trainer.py      # GNN training pipeline
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment variables
│   └── models/
│       ├── gcn_cmapss.pt      # Trained GCN weights
│       ├── gat_cmapss.pt      # Trained GAT weights
│       ├── fusion_cmapss_fd001.pt  # Fusion model weights
│       └── training_results.json   # Training metrics
│
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Global styles
│   │   ├── index.css          # Tailwind imports
│   │   ├── components/
│   │   │   ├── AuthPage.jsx           # Login/Register
│   │   │   ├── HealthDashboard.jsx    # Health overview
│   │   │   ├── SensorTimeSeries.jsx   # Sensor charts
│   │   │   ├── FailurePrediction.jsx  # RUL predictions
│   │   │   ├── GraphVisualization.jsx # Sensor graph
│   │   │   ├── MaintenanceLogs.jsx    # Log viewer
│   │   │   ├── AlertsPanel.jsx        # Alert management
│   │   │   ├── ComparisonDashboard.jsx # Model comparison
│   │   │   ├── OrganizationManager.jsx # Team management
│   │   │   └── Sidebar.jsx            # Navigation
│   │   ├── hooks/
│   │   │   └── useWebSocket.js        # WebSocket hook
│   │   └── components/ui/             # Shadcn components
│   ├── package.json
│   └── .env
│
├── data/
│   └── cmapss/
│       ├── train_FD001.txt    # NASA CMAPSS training data
│       ├── test_FD001.txt     # NASA CMAPSS test data
│       ├── RUL_FD001.txt      # Ground truth RUL
│       └── train_FD002-4.txt  # Extended datasets
│
├── memory/
│   └── PRD.md                 # Product requirements
│
└── README.md                  # This file
```

---

## 🏋️ Training the Model

### Quick Test (5 epochs, 10 engines)
```bash
cd /app/backend
python cmapss_trainer.py --test
```

### Full Training (25+ epochs)
```bash
python cmapss_trainer.py --epochs 25 --batch-size 32 --hidden 64
```

### GPU Training (when available)
```bash
python cmapss_trainer.py --epochs 100 --batch-size 64 --hidden 128
```

### Training Output
```
2026-02-15 07:57:46 - INFO - Device: cpu (or cuda)
2026-02-15 07:57:46 - INFO - Loading CMAPSS FD001 dataset...
2026-02-15 07:57:46 - INFO - Loaded train: 20631 rows
2026-02-15 07:57:46 - INFO - Train samples: 2869, Val samples: 717
...
2026-02-15 08:23:37 - INFO - Final Evaluation
2026-02-15 08:23:37 - INFO - GNN Fusion F1: 0.897
2026-02-15 08:23:37 - INFO - Threshold F1: 0.791
2026-02-15 08:23:37 - INFO - Improvement: 0.106
2026-02-15 08:23:37 - INFO - Annual ROI: 55.0%
2026-02-15 08:23:37 - INFO - Annual Savings: $134,393
```

### Extending to FD002-FD004
The datasets for FD002-FD004 (multiple operating conditions and fault modes) are already downloaded. To train:

```bash
# Modify cmapss_trainer.py to accept --dataset parameter
python cmapss_trainer.py --epochs 50 --dataset FD002
```

---

## 🔒 Role-Based Access Control

| Role | Permissions |
|------|-------------|
| **Admin** | Full access: manage org, users, machines, predictions, alerts, settings |
| **Operator** | Manage machines, run predictions, manage alerts, view reports |
| **Viewer** | Read-only: view dashboard, reports (no modifications) |

---

## 📈 ROI Calculation Model

The ROI calculator uses the following assumptions:

| Parameter | Default Value |
|-----------|---------------|
| Downtime cost per hour | $10,000 |
| Maintenance cost per intervention | $2,000 |
| Avg unplanned downtime | 8 hours |
| Avg planned downtime | 2 hours |
| False alarm action rate | 10% |

**Formula:**
```
Annual Savings = 
  (Threshold_Downtime_Cost - GNN_Downtime_Cost) +
  (Threshold_False_Alarm_Cost - GNN_False_Alarm_Cost) +
  Early_Warning_Savings

ROI % = (Annual_Savings / Threshold_Total_Cost) × 100
```

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
yarn test
```

### API Testing
```bash
# Health check
curl http://localhost:8001/api/

# Run full test suite
python -m pytest tests/ --cov=. --cov-report=html
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📊 Dataset

This project uses the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset for turbofan engine degradation simulation.

### Official Sources

| Source | Link |
|--------|------|
| **NASA Open Data Portal** | [https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) |
| **Direct Download (CMAPSSData.zip)** | [NASA Data Resource](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data/resource/5224bcd1-ad61-490b-93b9-2817288accb8) |
| **NASA Prognostics Center** | [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |

### Alternative Sources (Mirrors)

| Source | Link |
|--------|------|
| **Kaggle** | [https://www.kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) |
| **GitHub Mirror** | [https://github.com/hankroark/Turbofan-Engine-Degradation](https://github.com/hankroark/Turbofan-Engine-Degradation) |

### Dataset Description

The CMAPSS dataset contains run-to-failure simulations of turbofan engines under different operating conditions and fault modes:

| Dataset | Engines (Train) | Engines (Test) | Operating Conditions | Fault Modes |
|---------|-----------------|----------------|---------------------|-------------|
| **FD001** | 100 | 100 | 1 | 1 (HPC Degradation) |
| **FD002** | 260 | 259 | 6 | 1 (HPC Degradation) |
| **FD003** | 100 | 100 | 1 | 2 (HPC + Fan Degradation) |
| **FD004** | 249 | 248 | 6 | 2 (HPC + Fan Degradation) |

Each engine starts with different degrees of initial wear and manufacturing variation. The engine operates normally at the start and develops a fault at some point. The fault grows in magnitude until system failure.

### Data Format

Each row contains:
- `unit` - Engine unit number
- `cycle` - Time cycle
- `setting1-3` - Operational settings
- `s1-s21` - 21 sensor measurements

### Citation

If you use this dataset, please cite:
```
A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", 
International Conference on Prognostics and Health Management, 2008.
```

---

## 🙏 Acknowledgments

- **NASA** - CMAPSS Turbofan Engine Degradation Simulation Dataset
- **PyTorch Geometric** - Graph Neural Network framework
- **Shadcn/UI** - Beautiful React components
- **Recharts** - Composable charting library

---

## 📞 Support

For questions or support:
- 📧 Email: support@predictmaint.io
- 📖 Documentation: https://docs.predictmaint.io
- 🐛 Issues: https://github.com/your-org/predictmaint/issues

---

<p align="center">
  Built with ❤️ for Industrial IoT
</p>
