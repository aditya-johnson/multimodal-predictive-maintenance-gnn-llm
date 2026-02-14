# Multimodal Predictive Maintenance System - PRD

## Original Problem Statement
Build a Multimodal Predictive Maintenance system using GNN + LLM for industrial machines (turbines, motors, compressors, CNC machines). The system combines Graph Neural Networks, Large Language Model embeddings, and Time-series sensor modeling to predict machine failures, estimate Remaining Useful Life (RUL), detect anomalies early, and provide interpretable maintenance insights.

## User Choices
- **LLM Integration**: Local SentenceTransformers (all-MiniLM-L6-v2)
- **Data Source**: Both simulated + upload capability
- **GNN Implementation**: Full PyTorch Geometric ready (correlation-based graph)
- **Theme**: Dark industrial monitoring theme

## User Personas
1. **Industrial Engineers** - Monitor equipment health, analyze trends
2. **Maintenance Technicians** - Log observations, respond to alerts
3. **Plant Managers** - View dashboard summaries, plan maintenance schedules
4. **Researchers** - Study GNN/NLP models, evaluate predictions

## Core Requirements
- Equipment Health Dashboard with real-time gauges
- Sensor Time-Series visualization with interactive charts
- Failure Prediction panel with RUL estimates
- Graph Visualization of sensor correlations
- Maintenance Log Insights with NLP keyword extraction

## Architecture
- **Backend**: FastAPI + MongoDB
- **Frontend**: React + Tailwind CSS + Recharts + react-force-graph-2d
- **ML Stack**: PyTorch, SentenceTransformers, scipy, networkx, scikit-learn

## What's Been Implemented
- [x] Backend API with 15+ endpoints (Feb 14, 2026)
- [x] Data simulation engine with degradation patterns (Feb 14, 2026)
- [x] Correlation-based graph construction (Feb 14, 2026)
- [x] NLP risk keyword extraction (Feb 14, 2026)
- [x] Multimodal fusion prediction engine (Feb 14, 2026)
- [x] Dark industrial UI with 5 dashboard pages (Feb 14, 2026)
- [x] Health gauges, time-series charts, force-graph visualization (Feb 14, 2026)
- [x] CSV/JSON file upload capability (Feb 14, 2026)

## Prioritized Backlog
### P0 (Critical)
- All core features implemented ✓

### P1 (High Priority)
- Integrate actual PyTorch Geometric GCN/GAT model training
- Real-time WebSocket sensor streaming
- Email/SMS alerts for critical predictions

### P2 (Medium Priority)
- User authentication system
- Historical prediction accuracy tracking
- Export reports to PDF
- Mobile-responsive improvements

### P3 (Future)
- Dynamic graph (temporal changes)
- Comparison with baseline ML models
- Integration with industrial IoT platforms

## Next Tasks
1. Add PyTorch Geometric GCN model for actual graph learning
2. Implement real-time WebSocket updates for sensor data
3. Add alert notification system for critical health scores
4. Improve NLP with actual SentenceTransformers embeddings
