# Multimodal Predictive Maintenance System - PRD

## Original Problem Statement
Build a Multimodal Predictive Maintenance system using GNN + LLM for industrial machines (turbines, motors, compressors, CNC machines). The system combines Graph Neural Networks, Large Language Model embeddings, and Time-series sensor modeling to predict machine failures, estimate Remaining Useful Life (RUL), detect anomalies early, and provide interpretable maintenance insights.

## User Choices
- **LLM Integration**: Local SentenceTransformers (keyword extraction implemented)
- **Data Source**: Both simulated + upload capability
- **GNN Implementation**: Full PyTorch Geometric GCN + GAT models
- **Theme**: Dark industrial monitoring theme
- **Authentication**: JWT-based custom auth (email/password)
- **Dataset**: NASA CMAPSS-style data generated
- **Multi-tenancy**: Organizations model with RBAC

## User Personas
1. **Industrial Engineers** - Monitor equipment health, analyze trends
2. **Maintenance Technicians** - Log observations, respond to alerts
3. **Plant Managers** - View dashboard summaries, plan maintenance schedules
4. **Researchers** - Study GNN/NLP models, evaluate predictions

## Core Requirements
- [x] Equipment Health Dashboard with real-time gauges
- [x] Sensor Time-Series visualization with interactive charts
- [x] Failure Prediction panel with RUL estimates
- [x] Graph Visualization of sensor correlations
- [x] Maintenance Log Insights with NLP keyword extraction
- [x] User Authentication (JWT)
- [x] Multi-tenant data isolation
- [x] Real-time WebSocket streaming
- [x] Alert notification system
- [x] Role-Based Access Control (Admin/Operator/Viewer)
- [x] Organizations model for team management
- [x] PDF Report Generation

## Architecture
- **Backend**: FastAPI + MongoDB + PyTorch Geometric + JWT Auth
- **Frontend**: React + Tailwind CSS + Recharts + react-force-graph-2d
- **ML Stack**: PyTorch, GCN, GAT, scipy, networkx

## What's Been Implemented

### Version 4.0 (Feb 15, 2026)
- [x] Role-Based Access Control (RBAC) with 3 roles:
  - Admin: Full access (manage org, users, machines, predictions, alerts, settings)
  - Operator: Manage machines, run predictions, manage alerts, view reports
  - Viewer: Read-only access (view reports only)
- [x] Organizations model for enterprise multi-tenancy:
  - Create organizations
  - Switch between organizations
  - Invite members via email
  - Accept invitations
  - Update member roles
  - Remove members
- [x] PDF Report Generation with ReportLab:
  - Machine health summary
  - Failure predictions & RUL
  - Sensor trends (7/30/90 days)
  - Prediction history
  - Maintenance log insights
  - Recommended actions
  - Risk classification table
- [x] Frontend Organization Manager component
- [x] Sidebar navigation with Organization tab

### Version 3.0 (Feb 15, 2026)
- [x] JWT-based authentication (bcrypt + PyJWT)
- [x] User registration and login flows
- [x] Multi-tenant data isolation
- [x] Protected API routes with Bearer token
- [x] NASA CMAPSS-style training data generated
- [x] GNN model weight initialization saved

### Version 2.0 (Feb 14, 2026)
- [x] PyTorch Geometric GCN model (3-layer, 32 hidden channels)
- [x] PyTorch Geometric GAT model (4-head attention)
- [x] Ensemble prediction combining GCN + GAT
- [x] Real-time WebSocket streaming (5-second intervals)
- [x] Alert notification system with configurable thresholds
- [x] Alert Center UI with acknowledge workflow

### Version 1.0 (Feb 14, 2026)
- [x] Backend API with 25+ endpoints
- [x] Data simulation engine with degradation patterns
- [x] Correlation-based graph construction
- [x] NLP risk keyword extraction
- [x] Multimodal fusion prediction engine
- [x] Dark industrial UI with 7 dashboard pages

## Tech Stack Summary
| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| Database | MongoDB |
| Authentication | JWT (PyJWT + bcrypt) |
| GNN Models | PyTorch Geometric (GCN, GAT) |
| NLP | Risk keyword extraction |
| Real-time | WebSocket |
| Email | SendGrid (ready) |
| PDF Generation | ReportLab |
| Frontend | React 19 |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Graph Viz | react-force-graph-2d |
| Animations | Framer Motion |

## API Endpoints Summary
- **Auth**: `/api/auth/register`, `/api/auth/login`, `/api/auth/me`
- **Organizations**: CRUD, switch, invite, accept, members, roles
- **Machines**: CRUD + simulate + predict + report
- **Readings**: Get sensor data, upload CSV/JSON
- **Predictions**: Run GNN, get history
- **Logs**: Create, list with NLP analysis
- **Alerts**: List, acknowledge, settings
- **WebSocket**: `/ws/{machine_id}` for real-time streaming

## Prioritized Backlog

### P0 (Critical) - COMPLETED ✅
- All core features including RBAC, Organizations, PDF Reports

### P1 (High Priority)
- [ ] Train GNN on full NASA CMAPSS dataset (requires GPU)
- [ ] Historical prediction accuracy tracking

### P2 (Medium Priority)
- [ ] Enhanced PDF reports with charts
- [ ] Email notifications via SendGrid

### P3 (Future)
- [ ] Dynamic temporal graphs
- [ ] Baseline model comparison (LSTM, Random Forest)
- [ ] Industrial IoT integration (OPC-UA, MQTT)
- [ ] SMS alerts via Twilio
- [ ] Mobile-responsive improvements

## Next Tasks
1. Train GNN models on real NASA CMAPSS data with GPU
2. Configure SendGrid for email alerts
3. Add prediction accuracy tracking over time
4. Enhance PDF reports with embedded charts

## MOCKED Components
- **GNN Models**: Using initialized weights, not trained on real CMAPSS dataset
- **Sensor Data**: Simulated degradation patterns, not real IoT data
- **Email Alerts**: SendGrid integration present but requires API key
