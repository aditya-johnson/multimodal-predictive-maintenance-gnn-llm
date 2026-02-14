from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import numpy as np
import json
import io
import csv
from scipy import stats
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Multimodal Predictive Maintenance API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== SENDGRID CONFIG =====================
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'alerts@predictmaint.com')

# ===================== WEBSOCKET MANAGER =====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, machine_id: str):
        await websocket.accept()
        if machine_id not in self.active_connections:
            self.active_connections[machine_id] = []
        self.active_connections[machine_id].append(websocket)
        logger.info(f"WebSocket connected for machine {machine_id}")
    
    def disconnect(self, websocket: WebSocket, machine_id: str):
        if machine_id in self.active_connections:
            self.active_connections[machine_id].remove(websocket)
            logger.info(f"WebSocket disconnected for machine {machine_id}")
    
    async def broadcast(self, machine_id: str, data: dict):
        if machine_id in self.active_connections:
            for connection in self.active_connections[machine_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")
    
    async def broadcast_all(self, data: dict):
        for machine_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()

# ===================== GNN MODELS (PyTorch Geometric) =====================

class SensorGCN(nn.Module):
    """Graph Convolutional Network for sensor dependency modeling"""
    def __init__(self, num_features: int = 4, hidden_channels: int = 32, num_classes: int = 3):
        super(SensorGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index, edge_weight)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification layer
        x = self.lin(x)
        return F.softmax(x, dim=1)

class SensorGAT(nn.Module):
    """Graph Attention Network for sensor dependency modeling with attention"""
    def __init__(self, num_features: int = 4, hidden_channels: int = 32, num_heads: int = 4, num_classes: int = 3):
        super(SensorGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.3)
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification layer
        x = self.lin(x)
        return F.softmax(x, dim=1)

# Initialize GNN models
gcn_model = SensorGCN(num_features=4, hidden_channels=32, num_classes=3)
gat_model = SensorGAT(num_features=4, hidden_channels=32, num_heads=4, num_classes=3)

# Set models to evaluation mode (pre-trained weights would be loaded here in production)
gcn_model.eval()
gat_model.eval()

# ===================== PYDANTIC MODELS =====================

class Machine(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    machine_type: str
    location: str
    health_score: float = 100.0
    failure_probability: float = 0.0
    risk_level: str = "healthy"
    last_maintenance: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class MachineCreate(BaseModel):
    name: str
    machine_type: str
    location: str

class SensorReading(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    timestamp: str
    temperature: float
    pressure: float
    vibration: float
    rpm: float
    voltage: Optional[float] = None
    current: Optional[float] = None

class MaintenanceLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    timestamp: str
    log_text: str
    technician: str
    severity: str = "info"
    risk_keywords: List[str] = []
    embedding_similarity: Optional[float] = None

class MaintenanceLogCreate(BaseModel):
    machine_id: str
    log_text: str
    technician: str
    severity: str = "info"

class Prediction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    timestamp: str
    predicted_failure_date: str
    remaining_useful_life_days: float
    confidence_score: float
    failure_type: str
    gnn_score: float
    nlp_score: float
    fusion_score: float
    gcn_prediction: List[float] = []
    gat_prediction: List[float] = []

class Alert(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    machine_id: str
    machine_name: str
    alert_type: str
    severity: str
    message: str
    health_score: float
    failure_probability: float
    timestamp: str
    acknowledged: bool = False
    email_sent: bool = False

class AlertSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email_enabled: bool = True
    email_recipients: List[str] = []
    critical_threshold: float = 40.0
    warning_threshold: float = 70.0

class AlertSettingsUpdate(BaseModel):
    email_enabled: Optional[bool] = None
    email_recipients: Optional[List[str]] = None
    critical_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None

# ===================== EMAIL SERVICE =====================

async def send_alert_email(alert: Alert, recipients: List[str]):
    """Send email alert via SendGrid"""
    if not SENDGRID_API_KEY:
        logger.warning("SendGrid API key not configured, skipping email")
        return False
    
    if not recipients:
        logger.warning("No email recipients configured")
        return False
    
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        
        severity_color = {
            "critical": "#ef4444",
            "warning": "#facc15",
            "info": "#3b82f6"
        }.get(alert.severity, "#3b82f6")
        
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #1a1a1a; color: #e4e4e7; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: #27272a; border-radius: 8px; padding: 24px;">
                <h1 style="color: {severity_color}; margin-top: 0;">
                    ⚠️ {alert.alert_type.upper()} ALERT
                </h1>
                
                <div style="background-color: #18181b; border-left: 4px solid {severity_color}; padding: 16px; margin: 16px 0;">
                    <h2 style="margin: 0 0 8px 0; color: #e4e4e7;">{alert.machine_name}</h2>
                    <p style="margin: 0; color: #a1a1aa;">{alert.message}</p>
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
                    <tr>
                        <td style="padding: 8px; color: #a1a1aa;">Health Score:</td>
                        <td style="padding: 8px; color: {severity_color}; font-weight: bold; font-size: 24px;">
                            {alert.health_score:.1f}%
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #a1a1aa;">Failure Probability:</td>
                        <td style="padding: 8px; color: #e4e4e7; font-weight: bold;">
                            {alert.failure_probability:.1f}%
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; color: #a1a1aa;">Timestamp:</td>
                        <td style="padding: 8px; color: #e4e4e7;">
                            {alert.timestamp}
                        </td>
                    </tr>
                </table>
                
                <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #3f3f46;">
                    <p style="color: #71717a; font-size: 12px; margin: 0;">
                        This is an automated alert from PredictMaint - Multimodal Predictive Maintenance System.
                        <br>Please take immediate action for critical alerts.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        for recipient in recipients:
            message = Mail(
                from_email=Email(SENDER_EMAIL, "PredictMaint Alerts"),
                to_emails=To(recipient),
                subject=f"[{alert.severity.upper()}] {alert.machine_name} - {alert.alert_type}",
                html_content=Content("text/html", html_content)
            )
            
            response = sg.send(message)
            logger.info(f"Alert email sent to {recipient}, status: {response.status_code}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
        return False

# ===================== DATA SIMULATION =====================

def generate_degradation_pattern(days: int, failure_point: float = 0.8) -> np.ndarray:
    """Generate realistic degradation pattern."""
    x = np.linspace(0, 1, days)
    degradation = 1 - np.exp(-3 * (x ** 2))
    noise = np.random.normal(0, 0.02, days)
    return np.clip(degradation + noise, 0, 1)

def simulate_sensor_data(machine_id: str, days: int = 90) -> List[dict]:
    """Simulate run-to-failure sensor data."""
    readings = []
    degradation = generate_degradation_pattern(days)
    
    base_temp = 45 + np.random.uniform(-5, 5)
    base_pressure = 100 + np.random.uniform(-10, 10)
    base_vibration = 0.5 + np.random.uniform(-0.1, 0.1)
    base_rpm = 3000 + np.random.uniform(-100, 100)
    
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    
    for i in range(days * 24):
        d = degradation[min(i // 24, days - 1)]
        
        temp = base_temp + d * 35 + np.random.normal(0, 2)
        pressure = base_pressure - d * 25 + np.random.normal(0, 3)
        vibration = base_vibration + d * 4.5 + np.random.normal(0, 0.1)
        rpm = base_rpm - d * 500 + np.random.normal(0, 50)
        
        if np.random.random() < 0.02 * d:
            vibration *= 1.5
            temp += 10
        
        reading = {
            "id": str(uuid.uuid4()),
            "machine_id": machine_id,
            "timestamp": (start_time + timedelta(hours=i)).isoformat(),
            "temperature": round(float(temp), 2),
            "pressure": round(float(pressure), 2),
            "vibration": round(float(vibration), 3),
            "rpm": round(float(rpm), 1),
            "voltage": round(float(220 + np.random.normal(0, 5)), 1),
            "current": round(float(15 + d * 10 + np.random.normal(0, 1)), 2)
        }
        readings.append(reading)
    
    return readings

def generate_live_reading(machine_id: str, base_health: float) -> dict:
    """Generate a single live sensor reading based on machine health"""
    degradation = 1 - (base_health / 100)
    
    temp = 45 + degradation * 35 + np.random.normal(0, 2)
    pressure = 100 - degradation * 25 + np.random.normal(0, 3)
    vibration = 0.5 + degradation * 4.5 + np.random.normal(0, 0.1)
    rpm = 3000 - degradation * 500 + np.random.normal(0, 50)
    
    return {
        "id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(float(temp), 2),
        "pressure": round(float(pressure), 2),
        "vibration": round(float(vibration), 3),
        "rpm": round(float(rpm), 1),
        "voltage": round(float(220 + np.random.normal(0, 5)), 1),
        "current": round(float(15 + degradation * 10 + np.random.normal(0, 1)), 2)
    }

def calculate_health_score(readings: List[dict]) -> tuple:
    """Calculate health score from recent sensor readings."""
    if not readings:
        return 100.0, 0.0, "healthy"
    
    recent = readings[-24:]
    
    temps = [r["temperature"] for r in recent]
    vibs = [r["vibration"] for r in recent]
    pressures = [r["pressure"] for r in recent]
    
    temp_score = max(0, 100 - (np.mean(temps) - 45) * 2)
    vib_score = max(0, 100 - np.mean(vibs) * 20)
    pressure_score = max(0, min(100, np.mean(pressures)))
    
    health_score = (temp_score * 0.3 + vib_score * 0.4 + pressure_score * 0.3)
    health_score = max(0, min(100, health_score))
    
    failure_prob = max(0, min(100, (100 - health_score) * 1.2))
    
    if health_score >= 70:
        risk_level = "healthy"
    elif health_score >= 40:
        risk_level = "warning"
    else:
        risk_level = "critical"
    
    return round(health_score, 1), round(failure_prob, 1), risk_level

# ===================== GNN GRAPH CONSTRUCTION =====================

def build_pyg_graph(readings: List[dict]) -> Data:
    """Build PyTorch Geometric graph from sensor readings"""
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    
    if len(readings) < 10:
        # Default node features
        x = torch.tensor([
            [50.0, 100.0, 0.5, 3000.0],
            [50.0, 100.0, 0.5, 3000.0],
            [50.0, 100.0, 0.5, 3000.0],
            [50.0, 100.0, 0.5, 3000.0]
        ], dtype=torch.float)
        
        # Default edges (fully connected)
        edge_index = torch.tensor([
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3]
        ], dtype=torch.long)
        
        edge_weight = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    # Extract sensor data
    data = {s: [] for s in sensors}
    for r in readings[-500:]:
        for s in sensors:
            if s in r:
                data[s].append(r[s])
    
    # Build node features (normalized sensor statistics)
    node_features = []
    for s in sensors:
        if data[s]:
            mean_val = np.mean(data[s])
            std_val = np.std(data[s])
            min_val = np.min(data[s])
            max_val = np.max(data[s])
            node_features.append([mean_val, std_val, min_val, max_val])
        else:
            node_features.append([0.0, 0.0, 0.0, 0.0])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Normalize features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
    
    # Build edges based on correlations
    edges_src = []
    edges_dst = []
    edge_weights = []
    
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) and len(data[s1]) > 2:
                try:
                    corr, _ = stats.pearsonr(data[s1], data[s2])
                    if abs(corr) > 0.2:
                        edges_src.append(i)
                        edges_dst.append(j)
                        edge_weights.append(abs(corr))
                        # Add reverse edge for undirected graph
                        edges_src.append(j)
                        edges_dst.append(i)
                        edge_weights.append(abs(corr))
                except:
                    pass
    
    if not edges_src:
        # Fallback to fully connected
        for i in range(4):
            for j in range(i+1, 4):
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                edge_weights.extend([0.5, 0.5])
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

def build_sensor_correlation_graph(readings: List[dict]) -> Dict:
    """Build correlation graph for visualization"""
    if len(readings) < 10:
        return {
            "nodes": [
                {"id": "temperature", "group": 1, "value": 50},
                {"id": "pressure", "group": 2, "value": 50},
                {"id": "vibration", "group": 1, "value": 50},
                {"id": "rpm", "group": 2, "value": 50},
                {"id": "voltage", "group": 3, "value": 50},
                {"id": "current", "group": 3, "value": 50}
            ],
            "links": [
                {"source": "temperature", "target": "vibration", "weight": 0.5},
                {"source": "pressure", "target": "rpm", "weight": 0.5},
                {"source": "voltage", "target": "current", "weight": 0.8}
            ]
        }
    
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    data = {s: [] for s in sensors}
    
    for r in readings[-500:]:
        for s in sensors:
            if s in r:
                data[s].append(r[s])
    
    nodes = []
    links = []
    
    for i, s in enumerate(sensors):
        variance = np.var(data[s]) if data[s] else 0
        nodes.append({
            "id": s,
            "group": (i % 3) + 1,
            "value": min(100, variance * 10)
        })
    
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) and len(data[s1]) > 2:
                try:
                    corr, _ = stats.pearsonr(data[s1], data[s2])
                    if abs(corr) > 0.3:
                        links.append({
                            "source": s1,
                            "target": s2,
                            "weight": round(abs(corr), 3)
                        })
                except:
                    pass
    
    return {"nodes": nodes, "links": links}

# ===================== GNN PREDICTION =====================

def gnn_predict_pytorch(readings: List[dict]) -> Dict:
    """Run actual PyTorch Geometric GNN prediction"""
    graph_data = build_pyg_graph(readings)
    
    with torch.no_grad():
        # GCN prediction
        gcn_out = gcn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        gcn_probs = gcn_out.squeeze().tolist()
        
        # GAT prediction
        gat_out = gat_model(graph_data.x, graph_data.edge_index)
        gat_probs = gat_out.squeeze().tolist()
    
    # Classes: [healthy, warning, critical]
    # Ensemble prediction
    ensemble_probs = [(g + a) / 2 for g, a in zip(gcn_probs, gat_probs)]
    
    # Convert to risk score (higher = more risk)
    risk_score = ensemble_probs[1] * 0.3 + ensemble_probs[2] * 0.7
    
    return {
        "gcn_prediction": gcn_probs,
        "gat_prediction": gat_probs,
        "ensemble_prediction": ensemble_probs,
        "risk_score": float(risk_score),
        "predicted_class": ["healthy", "warning", "critical"][np.argmax(ensemble_probs)]
    }

# ===================== NLP PROCESSING =====================

RISK_KEYWORDS = [
    "abnormal", "noise", "leak", "leakage", "vibration", "excessive",
    "overheating", "failure", "broken", "crack", "worn", "degraded",
    "malfunction", "error", "warning", "critical", "urgent", "bearing",
    "motor", "seal", "belt", "corrosion", "fatigue", "alignment"
]

def analyze_maintenance_log(text: str) -> tuple:
    """Extract risk keywords and calculate risk score."""
    text_lower = text.lower()
    found_keywords = [kw for kw in RISK_KEYWORDS if kw in text_lower]
    risk_score = min(1.0, len(found_keywords) * 0.15)
    return found_keywords, risk_score

def nlp_predict(logs: List[dict]) -> float:
    """NLP-based failure prediction from maintenance logs."""
    if not logs:
        return 0.0
    
    recent_logs = logs[-10:]
    total_risk = 0
    
    for log in recent_logs:
        keywords = log.get("risk_keywords", [])
        severity = log.get("severity", "info")
        
        keyword_score = len(keywords) * 0.1
        severity_score = {"info": 0, "warning": 0.2, "error": 0.4, "critical": 0.6}.get(severity, 0)
        
        total_risk += keyword_score + severity_score
    
    return min(1.0, total_risk / len(recent_logs))

# ===================== MULTIMODAL FUSION =====================

def multimodal_fusion_predict(gnn_result: Dict, nlp_score: float, health_score: float) -> dict:
    """Fuse GNN and NLP predictions for final output."""
    gnn_score = gnn_result["risk_score"]
    
    # Weighted fusion
    fusion_score = 0.5 * gnn_score + 0.25 * nlp_score + 0.25 * (1 - health_score / 100)
    fusion_score = min(1.0, max(0.0, fusion_score))
    
    # Estimate remaining useful life
    if fusion_score < 0.2:
        rul_days = 90 + np.random.uniform(-10, 10)
        failure_type = "None predicted"
    elif fusion_score < 0.4:
        rul_days = 60 + np.random.uniform(-10, 10)
        failure_type = "Minor wear"
    elif fusion_score < 0.6:
        rul_days = 30 + np.random.uniform(-5, 5)
        failure_type = "Component degradation"
    elif fusion_score < 0.8:
        rul_days = 14 + np.random.uniform(-3, 3)
        failure_type = "Bearing failure likely"
    else:
        rul_days = 7 + np.random.uniform(-2, 2)
        failure_type = "Imminent failure"
    
    predicted_date = datetime.now(timezone.utc) + timedelta(days=rul_days)
    confidence = 0.6 + 0.3 * (1 - abs(fusion_score - 0.5) * 2)
    
    return {
        "predicted_failure_date": predicted_date.isoformat(),
        "remaining_useful_life_days": round(rul_days, 1),
        "confidence_score": round(confidence, 2),
        "failure_type": failure_type,
        "gnn_score": round(gnn_score, 3),
        "nlp_score": round(nlp_score, 3),
        "fusion_score": round(fusion_score, 3),
        "gcn_prediction": gnn_result["gcn_prediction"],
        "gat_prediction": gnn_result["gat_prediction"]
    }

# ===================== ALERT SYSTEM =====================

async def check_and_create_alert(machine: dict, health_score: float, failure_prob: float, risk_level: str):
    """Check thresholds and create alerts if needed"""
    settings = await db.alert_settings.find_one({}, {"_id": 0})
    
    if not settings:
        settings = {
            "email_enabled": True,
            "email_recipients": [],
            "critical_threshold": 40.0,
            "warning_threshold": 70.0
        }
    
    alert_type = None
    severity = None
    message = None
    
    if health_score < settings["critical_threshold"]:
        alert_type = "Critical Health"
        severity = "critical"
        message = f"Machine health score dropped to {health_score:.1f}%. Immediate attention required!"
    elif health_score < settings["warning_threshold"]:
        alert_type = "Health Warning"
        severity = "warning"
        message = f"Machine health score at {health_score:.1f}%. Schedule maintenance soon."
    
    if alert_type:
        alert = Alert(
            machine_id=machine["id"],
            machine_name=machine["name"],
            alert_type=alert_type,
            severity=severity,
            message=message,
            health_score=health_score,
            failure_probability=failure_prob,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Save alert
        await db.alerts.insert_one(alert.model_dump())
        
        # Broadcast to WebSocket
        await manager.broadcast_all({
            "type": "alert",
            "data": alert.model_dump()
        })
        
        # Send email if enabled
        if settings["email_enabled"] and settings["email_recipients"]:
            email_sent = await send_alert_email(alert, settings["email_recipients"])
            if email_sent:
                await db.alerts.update_one(
                    {"id": alert.id},
                    {"$set": {"email_sent": True}}
                )
        
        return alert
    
    return None

# ===================== API ENDPOINTS =====================

@api_router.get("/")
async def root():
    return {"message": "Multimodal Predictive Maintenance API", "status": "operational", "version": "2.0"}

# Machine endpoints
@api_router.post("/machines", response_model=Machine)
async def create_machine(input: MachineCreate):
    machine = Machine(**input.model_dump())
    doc = machine.model_dump()
    await db.machines.insert_one(doc)
    return machine

@api_router.get("/machines", response_model=List[Machine])
async def get_machines():
    machines = await db.machines.find({}, {"_id": 0}).to_list(100)
    return machines

@api_router.get("/machines/{machine_id}", response_model=Machine)
async def get_machine(machine_id: str):
    machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@api_router.delete("/machines/{machine_id}")
async def delete_machine(machine_id: str):
    result = await db.machines.delete_one({"id": machine_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Machine not found")
    await db.sensor_readings.delete_many({"machine_id": machine_id})
    await db.maintenance_logs.delete_many({"machine_id": machine_id})
    await db.predictions.delete_many({"machine_id": machine_id})
    return {"message": "Machine deleted"}

# Sensor reading endpoints
@api_router.get("/machines/{machine_id}/readings")
async def get_sensor_readings(machine_id: str, limit: int = 500):
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(limit)
    return readings[::-1]

@api_router.post("/machines/{machine_id}/simulate")
async def simulate_machine_data(machine_id: str, days: int = 90):
    """Simulate sensor data for a machine."""
    machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = simulate_sensor_data(machine_id, days)
    
    batch_size = 500
    for i in range(0, len(readings), batch_size):
        batch = readings[i:i+batch_size]
        await db.sensor_readings.insert_many(batch)
    
    health_score, failure_prob, risk_level = calculate_health_score(readings)
    await db.machines.update_one(
        {"id": machine_id},
        {"$set": {
            "health_score": health_score,
            "failure_probability": failure_prob,
            "risk_level": risk_level
        }}
    )
    
    # Check for alerts
    machine["health_score"] = health_score
    await check_and_create_alert(machine, health_score, failure_prob, risk_level)
    
    return {
        "message": f"Simulated {len(readings)} sensor readings",
        "health_score": health_score,
        "failure_probability": failure_prob,
        "risk_level": risk_level
    }

# Maintenance log endpoints
@api_router.post("/maintenance-logs", response_model=MaintenanceLog)
async def create_maintenance_log(input: MaintenanceLogCreate):
    risk_keywords, risk_score = analyze_maintenance_log(input.log_text)
    
    log = MaintenanceLog(
        machine_id=input.machine_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        log_text=input.log_text,
        technician=input.technician,
        severity=input.severity,
        risk_keywords=risk_keywords,
        embedding_similarity=risk_score
    )
    
    doc = log.model_dump()
    await db.maintenance_logs.insert_one(doc)
    return log

@api_router.get("/machines/{machine_id}/maintenance-logs", response_model=List[MaintenanceLog])
async def get_maintenance_logs(machine_id: str):
    logs = await db.maintenance_logs.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(100)
    return logs

# Graph endpoints
@api_router.get("/machines/{machine_id}/sensor-graph")
async def get_sensor_graph(machine_id: str):
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(500)
    
    graph = build_sensor_correlation_graph(readings[::-1])
    return graph

# Prediction endpoints
@api_router.post("/machines/{machine_id}/predict")
async def predict_failure(machine_id: str):
    """Run multimodal prediction with PyTorch Geometric GNN"""
    machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(1000)
    readings = readings[::-1]
    
    logs = await db.maintenance_logs.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).to_list(100)
    
    # PyTorch Geometric GNN prediction
    gnn_result = gnn_predict_pytorch(readings)
    
    # NLP prediction
    nlp_score = nlp_predict(logs)
    
    # Health score
    health_score, _, _ = calculate_health_score(readings)
    
    # Multimodal fusion
    prediction_data = multimodal_fusion_predict(gnn_result, nlp_score, health_score)
    
    prediction = Prediction(
        machine_id=machine_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **prediction_data
    )
    
    doc = prediction.model_dump()
    await db.predictions.insert_one(doc)
    
    # Update machine
    new_failure_prob = round(prediction_data["fusion_score"] * 100, 1)
    new_risk_level = "critical" if prediction_data["fusion_score"] > 0.6 else \
                     "warning" if prediction_data["fusion_score"] > 0.3 else "healthy"
    
    await db.machines.update_one(
        {"id": machine_id},
        {"$set": {
            "health_score": health_score,
            "failure_probability": new_failure_prob,
            "risk_level": new_risk_level
        }}
    )
    
    # Check alerts
    machine["name"] = machine.get("name", "Unknown")
    await check_and_create_alert(machine, health_score, new_failure_prob, new_risk_level)
    
    return prediction

@api_router.get("/machines/{machine_id}/predictions", response_model=List[Prediction])
async def get_predictions(machine_id: str):
    predictions = await db.predictions.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(50)
    return predictions

# Alert endpoints
@api_router.get("/alerts")
async def get_alerts(limit: int = 50, unacknowledged_only: bool = False):
    query = {"acknowledged": False} if unacknowledged_only else {}
    alerts = await db.alerts.find(query, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return alerts

@api_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    result = await db.alerts.update_one(
        {"id": alert_id},
        {"$set": {"acknowledged": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}

@api_router.get("/alert-settings")
async def get_alert_settings():
    settings = await db.alert_settings.find_one({}, {"_id": 0})
    if not settings:
        settings = AlertSettings().model_dump()
        await db.alert_settings.insert_one(settings)
    return settings

@api_router.put("/alert-settings")
async def update_alert_settings(update: AlertSettingsUpdate):
    update_dict = {k: v for k, v in update.model_dump().items() if v is not None}
    if update_dict:
        await db.alert_settings.update_one(
            {},
            {"$set": update_dict},
            upsert=True
        )
    settings = await db.alert_settings.find_one({}, {"_id": 0})
    return settings

# Data upload endpoint
@api_router.post("/upload")
async def upload_sensor_data(file: UploadFile = File(...)):
    """Upload sensor data from CSV/JSON file."""
    content = await file.read()
    
    try:
        if file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            readings = data if isinstance(data, list) else [data]
        elif file.filename.endswith('.csv'):
            reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
            readings = []
            for row in reader:
                reading = {
                    "id": str(uuid.uuid4()),
                    "machine_id": row.get("machine_id", "unknown"),
                    "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "temperature": float(row.get("temperature", row.get("temp", 0))),
                    "pressure": float(row.get("pressure", 0)),
                    "vibration": float(row.get("vibration", 0)),
                    "rpm": float(row.get("rpm", 0)),
                    "voltage": float(row.get("voltage", 0)) if row.get("voltage") else None,
                    "current": float(row.get("current", 0)) if row.get("current") else None
                }
                readings.append(reading)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or JSON.")
        
        if readings:
            await db.sensor_readings.insert_many(readings)
        
        return {"message": f"Uploaded {len(readings)} readings", "count": len(readings)}
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Dashboard summary endpoint
@api_router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get overall dashboard summary."""
    machines = await db.machines.find({}, {"_id": 0}).to_list(100)
    
    total_machines = len(machines)
    healthy = sum(1 for m in machines if m.get("risk_level") == "healthy")
    warning = sum(1 for m in machines if m.get("risk_level") == "warning")
    critical = sum(1 for m in machines if m.get("risk_level") == "critical")
    
    avg_health = np.mean([m.get("health_score", 100) for m in machines]) if machines else 100
    
    # Count unacknowledged alerts
    alert_count = await db.alerts.count_documents({"acknowledged": False})
    
    return {
        "total_machines": total_machines,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "average_health_score": round(avg_health, 1),
        "unacknowledged_alerts": alert_count,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

# Seed demo data endpoint
@api_router.post("/seed-demo")
async def seed_demo_data():
    """Seed database with demo machines and data."""
    await db.machines.delete_many({})
    await db.sensor_readings.delete_many({})
    await db.maintenance_logs.delete_many({})
    await db.predictions.delete_many({})
    await db.alerts.delete_many({})
    
    demo_machines = [
        {"name": "Turbine-A1", "machine_type": "Gas Turbine", "location": "Plant Floor A"},
        {"name": "Motor-B2", "machine_type": "Electric Motor", "location": "Assembly Line B"},
        {"name": "Compressor-C3", "machine_type": "Air Compressor", "location": "Utility Room C"},
        {"name": "CNC-D4", "machine_type": "CNC Machine", "location": "Machining Center D"},
        {"name": "Pump-E5", "machine_type": "Hydraulic Pump", "location": "Hydraulics Bay E"}
    ]
    
    demo_logs = [
        {"log_text": "Abnormal bearing noise detected during routine inspection", "technician": "John Smith", "severity": "warning"},
        {"log_text": "Slight oil leakage observed near main seal", "technician": "Maria Garcia", "severity": "warning"},
        {"log_text": "Excessive vibration reported by operator", "technician": "Robert Chen", "severity": "error"},
        {"log_text": "Regular maintenance completed, all systems normal", "technician": "Sarah Johnson", "severity": "info"},
        {"log_text": "Motor overheating detected, cooling system checked", "technician": "Mike Wilson", "severity": "critical"}
    ]
    
    created_machines = []
    for i, m_data in enumerate(demo_machines):
        machine = Machine(**m_data)
        await db.machines.insert_one(machine.model_dump())
        created_machines.append(machine)
        
        days = 30 + i * 15
        readings = simulate_sensor_data(machine.id, days)
        
        batch_size = 500
        for j in range(0, len(readings), batch_size):
            batch = readings[j:j+batch_size]
            await db.sensor_readings.insert_many(batch)
        
        health_score, failure_prob, risk_level = calculate_health_score(readings)
        await db.machines.update_one(
            {"id": machine.id},
            {"$set": {
                "health_score": health_score,
                "failure_probability": failure_prob,
                "risk_level": risk_level
            }}
        )
        
        log_data = demo_logs[i % len(demo_logs)]
        risk_keywords, risk_score = analyze_maintenance_log(log_data["log_text"])
        log = MaintenanceLog(
            machine_id=machine.id,
            timestamp=(datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 10))).isoformat(),
            log_text=log_data["log_text"],
            technician=log_data["technician"],
            severity=log_data["severity"],
            risk_keywords=risk_keywords,
            embedding_similarity=risk_score
        )
        await db.maintenance_logs.insert_one(log.model_dump())
    
    return {
        "message": "Demo data seeded successfully",
        "machines_created": len(created_machines),
        "machine_ids": [m.id for m in created_machines]
    }

# WebSocket endpoint for real-time streaming
@app.websocket("/ws/{machine_id}")
async def websocket_endpoint(websocket: WebSocket, machine_id: str):
    """WebSocket endpoint for real-time sensor streaming"""
    await manager.connect(websocket, machine_id)
    
    try:
        # Get machine info
        machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
        if not machine:
            await websocket.close(code=4004)
            return
        
        while True:
            # Generate live reading every 5 seconds
            reading = generate_live_reading(machine_id, machine.get("health_score", 50))
            
            # Save to database
            await db.sensor_readings.insert_one(reading)
            
            # Get recent readings for health calculation
            recent_readings = await db.sensor_readings.find(
                {"machine_id": machine_id}, {"_id": 0}
            ).sort("timestamp", -1).to_list(24)
            
            health_score, failure_prob, risk_level = calculate_health_score(recent_readings[::-1])
            
            # Update machine
            await db.machines.update_one(
                {"id": machine_id},
                {"$set": {
                    "health_score": health_score,
                    "failure_probability": failure_prob,
                    "risk_level": risk_level
                }}
            )
            
            # Check for alerts
            machine_updated = await db.machines.find_one({"id": machine_id}, {"_id": 0})
            await check_and_create_alert(machine_updated, health_score, failure_prob, risk_level)
            
            # Send data to client
            await websocket.send_json({
                "type": "sensor_update",
                "reading": reading,
                "health_score": health_score,
                "failure_probability": failure_prob,
                "risk_level": risk_level
            })
            
            # Update local machine health for next iteration
            machine["health_score"] = health_score
            
            await asyncio.sleep(5)  # 5 second interval
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, machine_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, machine_id)

# WebSocket for global alerts
@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    await manager.connect(websocket, "global_alerts")
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "global_alerts")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
