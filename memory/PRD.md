# Multimodal Predictive Maintenance System - PRD

## Original Problem Statement
Build a Multimodal Predictive Maintenance system using GNN + LLM for industrial machines. Combines Graph Neural Networks, LLM embeddings, and time-series modeling to predict failures and estimate RUL.

## What's Been Implemented

### Version 5.0 (Feb 15, 2026) - GPU-Ready Training & Comparison Dashboard
- [x] **NASA CMAPSS FD001 Dataset Integration**
  - Downloaded real turbofan engine degradation data (100 engines, 21 sensors)
  - Implemented preprocessing pipeline with sliding window graph construction
- [x] **GPU-Ready GNN Training Pipeline** (`/app/backend/cmapss_trainer.py`)
  - CUDA-compatible code (runs on CPU, GPU-ready)
  - GCN + GAT fusion model with multi-task learning (classification + RUL regression)
  - Threshold baseline for comparison
  - Automated metrics computation and ROI calculation
- [x] **Model Comparison Dashboard** (`/app/frontend/src/components/ComparisonDashboard.jsx`)
  - Accuracy comparison bar chart
  - Performance radar chart (5 metrics)
  - Early warning lead time visualization
  - False alerts comparison table
  - Interactive ROI calculator with sliders
- [x] **API Endpoint**: `/api/model-comparison` - Returns training results

### Previous Versions (v1-v4)
- RBAC with 3 roles (Admin/Operator/Viewer)
- Organizations model for multi-tenancy
- PDF Report Generation
- WebSocket real-time streaming
- JWT Authentication
- PyTorch Geometric GCN/GAT models
- Alert notification system

## GPU Training Instructions
```bash
# Quick CPU test (5 epochs, 10 engines)
cd /app/backend && python cmapss_trainer.py --test

# Full GPU training (requires CUDA)
python cmapss_trainer.py --epochs 100 --batch-size 64 --hidden 128

# Results saved to /app/backend/models/training_results.json
```

## Comparison Dashboard Metrics
| Metric | GNN Fusion | Threshold | Winner |
|--------|-----------|-----------|--------|
| Early Warning | 10.5 days | 0 days | GNN |
| Critical Detection | 100% | 0% | GNN |
| False Positive Rate | 23% | 0% | Threshold |
| Missed Failure Rate | 0% | 100% | GNN |

## MOCKED Components
- **GNN Models**: Trained on test mode (5 epochs). Full training requires GPU
- **Sensor Data**: Using NASA CMAPSS simulated data
- **Email Alerts**: SendGrid requires API key

## Next Tasks
1. Run full GPU training with 100+ epochs
2. Tune hyperparameters (window size, hidden channels)
3. Extend to FD002-FD004 datasets
4. Add prediction accuracy tracking over time

## Files
- `/app/backend/cmapss_trainer.py` - GPU-ready training script
- `/app/backend/models/training_results.json` - Training metrics
- `/app/frontend/src/components/ComparisonDashboard.jsx` - Comparison UI
- `/app/data/cmapss/` - NASA CMAPSS FD001 dataset
