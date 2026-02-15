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
- **Multi-tenancy**: Basic - users see only their own machines

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

## Architecture
- **Backend**: FastAPI + MongoDB + PyTorch Geometric + JWT Auth
- **Frontend**: React + Tailwind CSS + Recharts + react-force-graph-2d
- **ML Stack**: PyTorch, GCN, GAT, scipy, networkx

## What's Been Implemented

### Version 1.0 (Feb 14, 2026)
- [x] Backend API with 15+ endpoints
- [x] Data simulation engine with degradation patterns
- [x] Correlation-based graph construction
- [x] NLP risk keyword extraction
- [x] Multimodal fusion prediction engine
- [x] Dark industrial UI with 5 dashboard pages
- [x] Health gauges, time-series charts, force-graph visualization
- [x] CSV/JSON file upload capability

### Version 2.0 (Feb 14, 2026)
- [x] PyTorch Geometric GCN model (3-layer, 32 hidden channels)
- [x] PyTorch Geometric GAT model (4-head attention)
- [x] Ensemble prediction combining GCN + GAT
- [x] Real-time WebSocket streaming (5-second intervals)
- [x] Alert notification system with configurable thresholds
- [x] Alert Center UI with acknowledge workflow
- [x] SendGrid email integration (ready to enable)

### Version 3.0 (Feb 15, 2026)
- [x] JWT-based authentication (bcrypt + PyJWT)
- [x] User registration and login flows
- [x] Multi-tenant data isolation (user_id on all collections)
- [x] Protected API routes with Bearer token
- [x] Session persistence with localStorage
- [x] NASA CMAPSS-style training data generated
- [x] GNN model weight initialization saved
- [x] Comprehensive README documentation

## Tech Stack Summary
| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| Database | MongoDB |
| Authentication | JWT (PyJWT + bcrypt) |
| GNN Models | PyTorch Geometric (GCN, GAT) |
| NLP | Risk keyword extraction |
| Real-time | WebSocket |
| Email | SendGrid |
| Frontend | React 19 |
| Styling | Tailwind CSS |
| Charts | Recharts |
| Graph Viz | react-force-graph-2d |
| Animations | Framer Motion |

## API Endpoints Summary
- **Auth**: `/api/auth/register`, `/api/auth/login`, `/api/auth/me`
- **Machines**: CRUD + simulate + predict
- **Readings**: Get sensor data, upload CSV/JSON
- **Predictions**: Run GNN, get history
- **Logs**: Create, list with NLP analysis
- **Alerts**: List, acknowledge, settings
- **WebSocket**: `/ws/{machine_id}` for real-time streaming

## Prioritized Backlog

### P0 (Critical) - COMPLETED ✅
- All core features implemented

### P1 (High Priority) - COMPLETED ✅
- PyTorch Geometric GNN models
- Real-time WebSocket streaming
- Alert notification system
- JWT authentication
- Multi-tenant support

### P2 (Medium Priority)
- [ ] Train GNN on full NASA CMAPSS dataset (requires GPU)
- [ ] Historical prediction accuracy tracking
- [ ] Export reports to PDF
- [ ] Role-based access control (Admin/Operator/Viewer)

### P3 (Future)
- [ ] Dynamic temporal graphs
- [ ] Baseline model comparison (LSTM, Random Forest)
- [ ] Industrial IoT integration (OPC-UA, MQTT)
- [ ] SMS alerts via Twilio
- [ ] Mobile-responsive improvements

## Next Tasks
1. Deploy to production environment
2. Train GNN models on real NASA CMAPSS data with GPU
3. Add role-based access for team collaboration
4. Implement PDF report generation
5. Add prediction accuracy tracking over time

## Files Modified in v3.0
- `/app/backend/server.py` - Complete rewrite with JWT auth + multi-tenant
- `/app/backend/gnn_training.py` - GNN training script
- `/app/backend/data/train_FD001.csv` - CMAPSS-style training data
- `/app/backend/models/gcn_cmapss.pt` - GCN weights
- `/app/backend/models/gat_cmapss.pt` - GAT weights
- `/app/frontend/src/App.js` - Auth flow integration
- `/app/frontend/src/components/AuthPage.jsx` - Login/Register UI
- `/app/README.md` - Comprehensive documentation
