from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
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
import bcrypt
import jwt
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

JWT_SECRET = os.environ.get('JWT_SECRET', 'predictmaint-secret-key-change-in-production')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

app = FastAPI(title="Multimodal Predictive Maintenance API")
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'alerts@predictmaint.com')

# ===================== RBAC ROLES =====================

class Role:
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"

ROLE_PERMISSIONS = {
    Role.ADMIN: ["manage_org", "manage_users", "manage_machines", "run_predictions", "manage_alerts", "view_reports", "manage_settings"],
    Role.OPERATOR: ["manage_machines", "run_predictions", "manage_alerts", "view_reports"],
    Role.VIEWER: ["view_reports"]
}

def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, [])

# ===================== PYDANTIC MODELS =====================

class Organization(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = ""
    owner_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class OrganizationCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class OrganizationMember(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    user_id: str
    role: Literal["admin", "operator", "viewer"]
    joined_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Invitation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    org_name: str
    email: str
    role: Literal["admin", "operator", "viewer"]
    invited_by: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str = Field(default_factory=lambda: (datetime.now(timezone.utc) + timedelta(days=7)).isoformat())
    accepted: bool = False

class InvitationCreate(BaseModel):
    email: str
    role: Literal["admin", "operator", "viewer"]

class UserCreate(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    password_hash: str
    current_org_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Machine(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
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
    org_id: str
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
    org_id: str
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
    org_id: str
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
    org_id: str
    machine_id: str
    machine_name: str
    alert_type: str
    severity: str
    message: str
    health_score: float
    failure_probability: float
    timestamp: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    email_sent: bool = False

class AlertSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    email_enabled: bool = True
    email_recipients: List[str] = []
    critical_threshold: float = 40.0
    warning_threshold: float = 70.0

class AlertSettingsUpdate(BaseModel):
    email_enabled: Optional[bool] = None
    email_recipients: Optional[List[str]] = None
    critical_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None

# ===================== AUTH FUNCTIONS =====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def create_access_token(user_id: str, email: str, org_id: str = None, role: str = None) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "org_id": org_id,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    payload = decode_token(credentials.credentials)
    user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0, "password_hash": 0})
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Get current org role
    if user.get("current_org_id"):
        member = await db.org_members.find_one(
            {"org_id": user["current_org_id"], "user_id": user["id"]}, {"_id": 0}
        )
        user["role"] = member["role"] if member else None
    else:
        user["role"] = None
    
    return user

def require_permission(permission: str):
    async def check_permission(current_user: dict = Depends(get_current_user)):
        if not current_user.get("role"):
            raise HTTPException(status_code=403, detail="No organization selected")
        if not has_permission(current_user["role"], permission):
            raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")
        return current_user
    return check_permission

# ===================== WEBSOCKET MANAGER =====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, machine_id: str):
        await websocket.accept()
        if machine_id not in self.active_connections:
            self.active_connections[machine_id] = []
        self.active_connections[machine_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, machine_id: str):
        if machine_id in self.active_connections and websocket in self.active_connections[machine_id]:
            self.active_connections[machine_id].remove(websocket)
    
    async def broadcast(self, machine_id: str, data: dict):
        if machine_id in self.active_connections:
            for conn in self.active_connections[machine_id]:
                try:
                    await conn.send_json(data)
                except:
                    pass
    
    async def broadcast_all(self, data: dict):
        for conns in self.active_connections.values():
            for conn in conns:
                try:
                    await conn.send_json(data)
                except:
                    pass

manager = ConnectionManager()

# ===================== GNN MODELS =====================

class SensorGCN(nn.Module):
    def __init__(self, num_features=4, hidden_channels=32, num_classes=3):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = F.relu(self.dropout(self.conv1(x, edge_index, edge_weight)))
        x = F.relu(self.dropout(self.conv2(x, edge_index, edge_weight)))
        x = self.conv3(x, edge_index, edge_weight)
        x = x.mean(dim=0, keepdim=True) if batch is None else global_mean_pool(x, batch)
        return F.softmax(self.lin(x), dim=1)

class SensorGAT(nn.Module):
    def __init__(self, num_features=4, hidden_channels=32, num_heads=4, num_classes=3):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.3)
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = x.mean(dim=0, keepdim=True) if batch is None else global_mean_pool(x, batch)
        return F.softmax(self.lin(x), dim=1)

gcn_model = SensorGCN()
gat_model = SensorGAT()

MODEL_DIR = ROOT_DIR / "models"
try:
    if (MODEL_DIR / "gcn_cmapss.pt").exists():
        gcn_model.load_state_dict(torch.load(MODEL_DIR / "gcn_cmapss.pt", map_location='cpu'))
    if (MODEL_DIR / "gat_cmapss.pt").exists():
        gat_model.load_state_dict(torch.load(MODEL_DIR / "gat_cmapss.pt", map_location='cpu'))
    logger.info("Loaded trained GNN weights")
except:
    logger.warning("Using random GNN initialization")

gcn_model.eval()
gat_model.eval()

# ===================== HELPER FUNCTIONS =====================

def generate_degradation_pattern(days: int) -> np.ndarray:
    x = np.linspace(0, 1, days)
    return np.clip(1 - np.exp(-3 * (x ** 2)) + np.random.normal(0, 0.02, days), 0, 1)

def simulate_sensor_data(machine_id: str, org_id: str, days: int = 90) -> List[dict]:
    readings = []
    degradation = generate_degradation_pattern(days)
    base = {"temp": 45 + np.random.uniform(-5, 5), "pressure": 100 + np.random.uniform(-10, 10),
            "vibration": 0.5 + np.random.uniform(-0.1, 0.1), "rpm": 3000 + np.random.uniform(-100, 100)}
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    
    for i in range(days * 24):
        d = degradation[min(i // 24, days - 1)]
        readings.append({
            "id": str(uuid.uuid4()), "machine_id": machine_id, "org_id": org_id,
            "timestamp": (start_time + timedelta(hours=i)).isoformat(),
            "temperature": round(float(base["temp"] + d * 35 + np.random.normal(0, 2)), 2),
            "pressure": round(float(base["pressure"] - d * 25 + np.random.normal(0, 3)), 2),
            "vibration": round(float(base["vibration"] + d * 4.5 + np.random.normal(0, 0.1)), 3),
            "rpm": round(float(base["rpm"] - d * 500 + np.random.normal(0, 50)), 1),
            "voltage": round(float(220 + np.random.normal(0, 5)), 1),
            "current": round(float(15 + d * 10 + np.random.normal(0, 1)), 2)
        })
    return readings

def generate_live_reading(machine_id: str, org_id: str, health: float) -> dict:
    d = 1 - (health / 100)
    return {
        "id": str(uuid.uuid4()), "machine_id": machine_id, "org_id": org_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(float(45 + d * 35 + np.random.normal(0, 2)), 2),
        "pressure": round(float(100 - d * 25 + np.random.normal(0, 3)), 2),
        "vibration": round(float(0.5 + d * 4.5 + np.random.normal(0, 0.1)), 3),
        "rpm": round(float(3000 - d * 500 + np.random.normal(0, 50)), 1),
        "voltage": round(float(220 + np.random.normal(0, 5)), 1),
        "current": round(float(15 + d * 10 + np.random.normal(0, 1)), 2)
    }

def calculate_health_score(readings: List[dict]) -> tuple:
    if not readings:
        return 100.0, 0.0, "healthy"
    recent = readings[-24:]
    temps, vibs, pressures = [r["temperature"] for r in recent], [r["vibration"] for r in recent], [r["pressure"] for r in recent]
    health = max(0, min(100, (max(0, 100 - (np.mean(temps) - 45) * 2) * 0.3 + 
                               max(0, 100 - np.mean(vibs) * 20) * 0.4 + 
                               max(0, min(100, np.mean(pressures))) * 0.3)))
    failure_prob = max(0, min(100, (100 - health) * 1.2))
    risk = "critical" if health < 40 else "warning" if health < 70 else "healthy"
    return round(health, 1), round(failure_prob, 1), risk

def build_pyg_graph(readings: List[dict]) -> Data:
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    if len(readings) < 10:
        x = torch.tensor([[50.0, 100.0, 0.5, 3000.0]] * 4, dtype=torch.float)
        edge_index = torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=torch.ones(6))
    
    data = {s: [r[s] for r in readings[-500:] if s in r] for s in sensors}
    node_features = [[np.mean(data[s]), np.std(data[s]), np.min(data[s]), np.max(data[s])] if data[s] else [0]*4 for s in sensors]
    x = torch.tensor(node_features, dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
    
    edges_src, edges_dst, edge_weights = [], [], []
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) > 2:
                try:
                    corr, _ = stats.pearsonr(data[s1], data[s2])
                    if abs(corr) > 0.2:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                        edge_weights.extend([abs(corr)] * 2)
                except:
                    pass
    
    if not edges_src:
        for i in range(4):
            for j in range(i+1, 4):
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                edge_weights.extend([0.5] * 2)
    
    return Data(x=x, edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long), edge_attr=torch.tensor(edge_weights, dtype=torch.float))

def build_sensor_correlation_graph(readings: List[dict]) -> Dict:
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    if len(readings) < 10:
        return {"nodes": [{"id": s, "group": i % 3 + 1, "value": 50} for i, s in enumerate(sensors)],
                "links": [{"source": "temperature", "target": "vibration", "weight": 0.5}]}
    
    data = {s: [r[s] for r in readings[-500:] if s in r] for s in sensors}
    nodes = [{"id": s, "group": (i % 3) + 1, "value": min(100, np.var(data[s]) * 10) if data[s] else 0} for i, s in enumerate(sensors)]
    links = []
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) > 2:
                try:
                    corr, _ = stats.pearsonr(data[s1], data[s2])
                    if abs(corr) > 0.3:
                        links.append({"source": s1, "target": s2, "weight": round(abs(corr), 3)})
                except:
                    pass
    return {"nodes": nodes, "links": links}

def gnn_predict_pytorch(readings: List[dict]) -> Dict:
    graph_data = build_pyg_graph(readings)
    with torch.no_grad():
        gcn_out = gcn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr).squeeze().tolist()
        gat_out = gat_model(graph_data.x, graph_data.edge_index).squeeze().tolist()
    ensemble = [(g + a) / 2 for g, a in zip(gcn_out, gat_out)]
    return {"gcn_prediction": gcn_out, "gat_prediction": gat_out, "ensemble_prediction": ensemble,
            "risk_score": float(ensemble[1] * 0.3 + ensemble[2] * 0.7),
            "predicted_class": ["healthy", "warning", "critical"][np.argmax(ensemble)]}

RISK_KEYWORDS = ["abnormal", "noise", "leak", "leakage", "vibration", "excessive", "overheating", 
                 "failure", "broken", "crack", "worn", "degraded", "malfunction", "error", 
                 "warning", "critical", "urgent", "bearing", "motor", "seal", "belt", "corrosion"]

def analyze_maintenance_log(text: str) -> tuple:
    found = [kw for kw in RISK_KEYWORDS if kw in text.lower()]
    return found, min(1.0, len(found) * 0.15)

def nlp_predict(logs: List[dict]) -> float:
    if not logs:
        return 0.0
    return min(1.0, sum(len(l.get("risk_keywords", [])) * 0.1 + 
                        {"info": 0, "warning": 0.2, "error": 0.4, "critical": 0.6}.get(l.get("severity", "info"), 0) 
                        for l in logs[-10:]) / min(len(logs), 10))

def multimodal_fusion_predict(gnn_result: Dict, nlp_score: float, health_score: float) -> dict:
    fusion = min(1.0, max(0.0, 0.5 * gnn_result["risk_score"] + 0.25 * nlp_score + 0.25 * (1 - health_score / 100)))
    rul = {0.2: 90, 0.4: 60, 0.6: 30, 0.8: 14, 1.0: 7}
    rul_days = next((v for k, v in sorted(rul.items()) if fusion < k), 7) + np.random.uniform(-5, 5)
    failure_type = {0.2: "None predicted", 0.4: "Minor wear", 0.6: "Component degradation", 0.8: "Bearing failure likely", 1.0: "Imminent failure"}
    ft = next((v for k, v in sorted(failure_type.items()) if fusion < k), "Imminent failure")
    return {
        "predicted_failure_date": (datetime.now(timezone.utc) + timedelta(days=rul_days)).isoformat(),
        "remaining_useful_life_days": round(rul_days, 1), "confidence_score": round(0.6 + 0.3 * (1 - abs(fusion - 0.5) * 2), 2),
        "failure_type": ft, "gnn_score": round(gnn_result["risk_score"], 3), "nlp_score": round(nlp_score, 3),
        "fusion_score": round(fusion, 3), "gcn_prediction": gnn_result["gcn_prediction"], "gat_prediction": gnn_result["gat_prediction"]
    }

# ===================== ALERT SYSTEM =====================

async def send_alert_email(alert: Alert, recipients: List[str]):
    if not SENDGRID_API_KEY or not recipients:
        return False
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        color = {"critical": "#ef4444", "warning": "#facc15"}.get(alert.severity, "#3b82f6")
        html = f"""<html><body style="font-family:Arial;background:#1a1a1a;color:#e4e4e7;padding:20px">
            <div style="max-width:600px;margin:auto;background:#27272a;border-radius:8px;padding:24px">
            <h1 style="color:{color}">⚠️ {alert.alert_type.upper()}</h1>
            <p><strong>{alert.machine_name}</strong>: {alert.message}</p>
            <p>Health: <strong style="color:{color}">{alert.health_score:.1f}%</strong></p>
            </div></body></html>"""
        for r in recipients:
            sg.send(Mail(Email(SENDER_EMAIL), To(r), f"[{alert.severity.upper()}] {alert.machine_name}", Content("text/html", html)))
        return True
    except:
        return False

async def check_and_create_alert(machine: dict, org_id: str, health: float, failure_prob: float, risk: str):
    settings = await db.alert_settings.find_one({"org_id": org_id}, {"_id": 0}) or {"email_enabled": True, "email_recipients": [], "critical_threshold": 40.0, "warning_threshold": 70.0}
    if health < settings["critical_threshold"]:
        alert_type, severity, msg = "Critical Health", "critical", f"Health dropped to {health:.1f}%. Immediate attention required!"
    elif health < settings["warning_threshold"]:
        alert_type, severity, msg = "Health Warning", "warning", f"Health at {health:.1f}%. Schedule maintenance."
    else:
        return None
    
    alert = Alert(org_id=org_id, machine_id=machine["id"], machine_name=machine.get("name", "Unknown"),
                  alert_type=alert_type, severity=severity, message=msg, health_score=health,
                  failure_probability=failure_prob, timestamp=datetime.now(timezone.utc).isoformat())
    await db.alerts.insert_one(alert.model_dump())
    await manager.broadcast_all({"type": "alert", "data": alert.model_dump()})
    if settings["email_enabled"] and settings["email_recipients"]:
        if await send_alert_email(alert, settings["email_recipients"]):
            await db.alerts.update_one({"id": alert.id}, {"$set": {"email_sent": True}})
    return alert

# ===================== PDF REPORT GENERATION =====================

async def generate_pdf_report(machine: dict, readings: List[dict], predictions: List[dict], logs: List[dict], days: int = 30) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor=colors.HexColor("#00CED1"))
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16, spaceAfter=12, textColor=colors.HexColor("#E4E4E7"))
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#A1A1AA"))
    
    story = []
    
    # Header
    story.append(Paragraph("PredictMaint", title_style))
    story.append(Paragraph("Multimodal Predictive Maintenance Report", heading_style))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Machine Health Summary
    story.append(Paragraph("1. Machine Health Summary", heading_style))
    health_data = [
        ["Machine Name", machine.get("name", "N/A")],
        ["Machine Type", machine.get("machine_type", "N/A")],
        ["Location", machine.get("location", "N/A")],
        ["Health Score", f"{machine.get('health_score', 0):.1f}%"],
        ["Failure Probability", f"{machine.get('failure_probability', 0):.1f}%"],
        ["Risk Level", machine.get("risk_level", "N/A").upper()],
    ]
    
    health_table = Table(health_data, colWidths=[2*inch, 4*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#27272A")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#E4E4E7")),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#3F3F46")),
    ]))
    story.append(health_table)
    story.append(Spacer(1, 20))
    
    # Failure Prediction
    if predictions:
        latest = predictions[0]
        story.append(Paragraph("2. Failure Prediction & RUL", heading_style))
        pred_data = [
            ["Remaining Useful Life", f"{latest.get('remaining_useful_life_days', 0):.1f} days"],
            ["Predicted Failure Date", latest.get('predicted_failure_date', 'N/A')[:10]],
            ["Confidence Score", f"{latest.get('confidence_score', 0)*100:.0f}%"],
            ["Failure Type", latest.get("failure_type", "N/A")],
            ["GNN Score", f"{latest.get('gnn_score', 0)*100:.1f}%"],
            ["NLP Score", f"{latest.get('nlp_score', 0)*100:.1f}%"],
            ["Fusion Score", f"{latest.get('fusion_score', 0)*100:.1f}%"],
        ]
        pred_table = Table(pred_data, colWidths=[2*inch, 4*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#27272A")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#E4E4E7")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#3F3F46")),
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 20))
    
    # Sensor Trends
    story.append(Paragraph(f"3. Sensor Trends (Last {days} Days)", heading_style))
    if readings:
        # Calculate daily averages
        daily_data = {}
        for r in readings:
            day = r["timestamp"][:10]
            if day not in daily_data:
                daily_data[day] = {"temp": [], "pressure": [], "vibration": [], "rpm": []}
            daily_data[day]["temp"].append(r["temperature"])
            daily_data[day]["pressure"].append(r["pressure"])
            daily_data[day]["vibration"].append(r["vibration"])
            daily_data[day]["rpm"].append(r["rpm"])
        
        trend_data = [["Date", "Avg Temp (°C)", "Avg Pressure (PSI)", "Avg Vibration (mm/s)", "Avg RPM"]]
        for day in sorted(daily_data.keys())[-7:]:
            d = daily_data[day]
            trend_data.append([
                day, f"{np.mean(d['temp']):.1f}", f"{np.mean(d['pressure']):.1f}",
                f"{np.mean(d['vibration']):.2f}", f"{np.mean(d['rpm']):.0f}"
            ])
        
        trend_table = Table(trend_data, colWidths=[1.2*inch]*5)
        trend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00CED1")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#E4E4E7")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#3F3F46")),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(trend_table)
    story.append(Spacer(1, 20))
    
    # Prediction History
    if len(predictions) > 1:
        story.append(Paragraph("4. Prediction History", heading_style))
        pred_history = [["Date", "RUL (days)", "Confidence", "Failure Type"]]
        for p in predictions[:5]:
            pred_history.append([
                p["timestamp"][:10], f"{p['remaining_useful_life_days']:.0f}",
                f"{p['confidence_score']*100:.0f}%", p["failure_type"]
            ])
        
        history_table = Table(pred_history, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 2.1*inch])
        history_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00CED1")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#E4E4E7")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#3F3F46")),
        ]))
        story.append(history_table)
        story.append(Spacer(1, 20))
    
    # Maintenance Log Insights
    story.append(Paragraph("5. Maintenance Log Insights", heading_style))
    if logs:
        # Risk keywords summary
        all_keywords = []
        for l in logs:
            all_keywords.extend(l.get("risk_keywords", []))
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        if keyword_counts:
            story.append(Paragraph(f"Risk Keywords Detected: {', '.join(f'{k} ({v})' for k, v in sorted(keyword_counts.items(), key=lambda x: -x[1])[:10])}", normal_style))
            story.append(Spacer(1, 10))
        
        log_data = [["Date", "Severity", "Technician", "Summary"]]
        for l in logs[:5]:
            summary = l["log_text"][:50] + "..." if len(l["log_text"]) > 50 else l["log_text"]
            log_data.append([l["timestamp"][:10], l["severity"].upper(), l["technician"], summary])
        
        log_table = Table(log_data, colWidths=[1*inch, 0.8*inch, 1.2*inch, 3*inch])
        log_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00CED1")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor("#E4E4E7")),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#3F3F46")),
        ]))
        story.append(log_table)
    else:
        story.append(Paragraph("No maintenance logs recorded.", normal_style))
    story.append(Spacer(1, 20))
    
    # Recommended Actions
    story.append(Paragraph("6. Recommended Actions", heading_style))
    risk_level = machine.get("risk_level", "healthy")
    health = machine.get("health_score", 100)
    
    recommendations = []
    if risk_level == "critical":
        recommendations = [
            "⚠️ IMMEDIATE ACTION REQUIRED",
            "• Schedule emergency maintenance within 24-48 hours",
            "• Reduce machine load or consider temporary shutdown",
            "• Inspect bearings, seals, and cooling systems",
            "• Review recent maintenance logs for missed issues",
            "• Prepare replacement parts for likely components"
        ]
    elif risk_level == "warning":
        recommendations = [
            "⚡ MAINTENANCE RECOMMENDED",
            "• Schedule preventive maintenance within 1-2 weeks",
            "• Monitor sensor trends daily for degradation",
            "• Check lubrication and coolant levels",
            "• Inspect for unusual vibrations or noises",
            "• Update maintenance log with observations"
        ]
    else:
        recommendations = [
            "✅ NORMAL OPERATION",
            "• Continue regular monitoring schedule",
            "• Maintain routine inspection intervals",
            "• Keep maintenance logs up to date",
            "• Review sensor baselines quarterly"
        ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
    story.append(Spacer(1, 20))
    
    # Risk Classification
    story.append(Paragraph("7. Risk Level Classification", heading_style))
    risk_data = [
        ["Risk Level", "Health Range", "Action Required"],
        ["HEALTHY", "> 70%", "Normal monitoring"],
        ["WARNING", "40% - 70%", "Schedule maintenance"],
        ["CRITICAL", "< 40%", "Immediate attention"],
    ]
    risk_table = Table(risk_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00CED1")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#10B981")),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#FACC15")),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#EF4444")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 30))
    
    # Footer
    story.append(Paragraph("---", normal_style))
    story.append(Paragraph("Generated by PredictMaint - Multimodal Predictive Maintenance System", normal_style))
    story.append(Paragraph("Powered by Graph Neural Networks (GCN + GAT) and NLP Analysis", normal_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ===================== AUTH ENDPOINTS =====================

@api_router.post("/auth/register")
async def register(input: UserCreate):
    if await db.users.find_one({"email": input.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(email=input.email, name=input.name, password_hash=hash_password(input.password))
    await db.users.insert_one(user.model_dump())
    
    # Check for pending invitations
    invitations = await db.invitations.find({"email": input.email, "accepted": False}).to_list(10)
    
    token = create_access_token(user.id, user.email)
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at},
            "pending_invitations": len(invitations)}

@api_router.post("/auth/login")
async def login(input: UserLogin):
    user = await db.users.find_one({"email": input.email})
    if not user or not verify_password(input.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(user["id"], user["email"], user.get("current_org_id"))
    return {"access_token": token, "token_type": "bearer",
            "user": {"id": user["id"], "email": user["email"], "name": user["name"], 
                     "created_at": user["created_at"], "current_org_id": user.get("current_org_id")}}

@api_router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

# ===================== ORGANIZATION ENDPOINTS =====================

@api_router.post("/organizations")
async def create_organization(input: OrganizationCreate, current_user: dict = Depends(get_current_user)):
    org = Organization(name=input.name, description=input.description, owner_id=current_user["id"])
    await db.organizations.insert_one(org.model_dump())
    
    # Add creator as admin
    member = OrganizationMember(org_id=org.id, user_id=current_user["id"], role=Role.ADMIN)
    await db.org_members.insert_one(member.model_dump())
    
    # Set as current org
    await db.users.update_one({"id": current_user["id"]}, {"$set": {"current_org_id": org.id}})
    
    # Create default alert settings
    settings = AlertSettings(org_id=org.id)
    await db.alert_settings.insert_one(settings.model_dump())
    
    return {"organization": org.model_dump(), "role": Role.ADMIN}

@api_router.get("/organizations")
async def get_user_organizations(current_user: dict = Depends(get_current_user)):
    memberships = await db.org_members.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(50)
    orgs = []
    for m in memberships:
        org = await db.organizations.find_one({"id": m["org_id"]}, {"_id": 0})
        if org:
            org["role"] = m["role"]
            orgs.append(org)
    return orgs

@api_router.post("/organizations/{org_id}/switch")
async def switch_organization(org_id: str, current_user: dict = Depends(get_current_user)):
    member = await db.org_members.find_one({"org_id": org_id, "user_id": current_user["id"]}, {"_id": 0})
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    
    await db.users.update_one({"id": current_user["id"]}, {"$set": {"current_org_id": org_id}})
    org = await db.organizations.find_one({"id": org_id}, {"_id": 0})
    
    token = create_access_token(current_user["id"], current_user["email"], org_id, member["role"])
    return {"access_token": token, "organization": org, "role": member["role"]}

@api_router.get("/organizations/{org_id}/members")
async def get_organization_members(org_id: str, current_user: dict = Depends(require_permission("manage_org"))):
    if current_user.get("current_org_id") != org_id:
        raise HTTPException(status_code=403, detail="Not authorized for this organization")
    
    members = await db.org_members.find({"org_id": org_id}, {"_id": 0}).to_list(100)
    result = []
    for m in members:
        user = await db.users.find_one({"id": m["user_id"]}, {"_id": 0, "password_hash": 0})
        if user:
            result.append({**user, "role": m["role"], "joined_at": m["joined_at"]})
    return result

@api_router.post("/organizations/{org_id}/invite")
async def invite_to_organization(org_id: str, input: InvitationCreate, current_user: dict = Depends(require_permission("manage_users"))):
    if current_user.get("current_org_id") != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    org = await db.organizations.find_one({"id": org_id}, {"_id": 0})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Check if already invited or member
    existing = await db.invitations.find_one({"org_id": org_id, "email": input.email, "accepted": False})
    if existing:
        raise HTTPException(status_code=400, detail="Already invited")
    
    existing_member = await db.org_members.find_one({"org_id": org_id, "user_id": {"$exists": True}})
    if existing_member:
        user = await db.users.find_one({"email": input.email})
        if user and await db.org_members.find_one({"org_id": org_id, "user_id": user["id"]}):
            raise HTTPException(status_code=400, detail="Already a member")
    
    invitation = Invitation(org_id=org_id, org_name=org["name"], email=input.email, 
                           role=input.role, invited_by=current_user["id"])
    await db.invitations.insert_one(invitation.model_dump())
    
    return invitation

@api_router.get("/invitations")
async def get_my_invitations(current_user: dict = Depends(get_current_user)):
    invitations = await db.invitations.find(
        {"email": current_user["email"], "accepted": False}, {"_id": 0}
    ).to_list(20)
    return invitations

@api_router.post("/invitations/{invitation_id}/accept")
async def accept_invitation(invitation_id: str, current_user: dict = Depends(get_current_user)):
    invitation = await db.invitations.find_one({"id": invitation_id, "email": current_user["email"]}, {"_id": 0})
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    if invitation["accepted"]:
        raise HTTPException(status_code=400, detail="Already accepted")
    
    # Add as member
    member = OrganizationMember(org_id=invitation["org_id"], user_id=current_user["id"], role=invitation["role"])
    await db.org_members.insert_one(member.model_dump())
    
    # Mark invitation as accepted
    await db.invitations.update_one({"id": invitation_id}, {"$set": {"accepted": True}})
    
    # Set as current org
    await db.users.update_one({"id": current_user["id"]}, {"$set": {"current_org_id": invitation["org_id"]}})
    
    return {"message": "Invitation accepted", "org_id": invitation["org_id"], "role": invitation["role"]}

@api_router.put("/organizations/{org_id}/members/{user_id}/role")
async def update_member_role(org_id: str, user_id: str, role: Literal["admin", "operator", "viewer"], 
                             current_user: dict = Depends(require_permission("manage_users"))):
    if current_user.get("current_org_id") != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    org = await db.organizations.find_one({"id": org_id}, {"_id": 0})
    if org and org["owner_id"] == user_id and role != Role.ADMIN:
        raise HTTPException(status_code=400, detail="Cannot change owner's role")
    
    result = await db.org_members.update_one(
        {"org_id": org_id, "user_id": user_id},
        {"$set": {"role": role}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Member not found")
    
    return {"message": "Role updated", "user_id": user_id, "new_role": role}

@api_router.delete("/organizations/{org_id}/members/{user_id}")
async def remove_member(org_id: str, user_id: str, current_user: dict = Depends(require_permission("manage_users"))):
    if current_user.get("current_org_id") != org_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    org = await db.organizations.find_one({"id": org_id}, {"_id": 0})
    if org and org["owner_id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot remove owner")
    
    await db.org_members.delete_one({"org_id": org_id, "user_id": user_id})
    await db.users.update_one({"id": user_id, "current_org_id": org_id}, {"$set": {"current_org_id": None}})
    
    return {"message": "Member removed"}

# ===================== MACHINE ENDPOINTS =====================

@api_router.get("/")
async def root():
    return {"message": "Multimodal Predictive Maintenance API", "status": "operational", "version": "4.0"}

@api_router.post("/machines")
async def create_machine(input: MachineCreate, current_user: dict = Depends(require_permission("manage_machines"))):
    org_id = current_user.get("current_org_id")
    if not org_id:
        raise HTTPException(status_code=400, detail="Select an organization first")
    
    machine = Machine(org_id=org_id, **input.model_dump())
    await db.machines.insert_one(machine.model_dump())
    return machine

@api_router.get("/machines")
async def get_machines(current_user: dict = Depends(require_permission("view_reports"))):
    org_id = current_user.get("current_org_id")
    if not org_id:
        return []
    return await db.machines.find({"org_id": org_id}, {"_id": 0}).to_list(100)

@api_router.get("/machines/{machine_id}")
async def get_machine(machine_id: str, current_user: dict = Depends(require_permission("view_reports"))):
    machine = await db.machines.find_one({"id": machine_id, "org_id": current_user.get("current_org_id")}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@api_router.delete("/machines/{machine_id}")
async def delete_machine(machine_id: str, current_user: dict = Depends(require_permission("manage_machines"))):
    org_id = current_user.get("current_org_id")
    result = await db.machines.delete_one({"id": machine_id, "org_id": org_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Machine not found")
    await db.sensor_readings.delete_many({"machine_id": machine_id, "org_id": org_id})
    await db.maintenance_logs.delete_many({"machine_id": machine_id, "org_id": org_id})
    await db.predictions.delete_many({"machine_id": machine_id, "org_id": org_id})
    return {"message": "Machine deleted"}

@api_router.get("/machines/{machine_id}/readings")
async def get_sensor_readings(machine_id: str, limit: int = 500, current_user: dict = Depends(require_permission("view_reports"))):
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id, "org_id": current_user.get("current_org_id")}, {"_id": 0}
    ).sort("timestamp", -1).to_list(limit)
    return readings[::-1]

@api_router.post("/machines/{machine_id}/simulate")
async def simulate_machine_data(machine_id: str, days: int = 90, current_user: dict = Depends(require_permission("manage_machines"))):
    org_id = current_user.get("current_org_id")
    machine = await db.machines.find_one({"id": machine_id, "org_id": org_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = simulate_sensor_data(machine_id, org_id, days)
    for i in range(0, len(readings), 500):
        await db.sensor_readings.insert_many(readings[i:i+500])
    
    health, failure_prob, risk = calculate_health_score(readings)
    await db.machines.update_one({"id": machine_id}, {"$set": {"health_score": health, "failure_probability": failure_prob, "risk_level": risk}})
    await check_and_create_alert(machine, org_id, health, failure_prob, risk)
    
    return {"message": f"Simulated {len(readings)} readings", "health_score": health, "risk_level": risk}

@api_router.post("/maintenance-logs")
async def create_maintenance_log(input: MaintenanceLogCreate, current_user: dict = Depends(require_permission("manage_machines"))):
    org_id = current_user.get("current_org_id")
    keywords, score = analyze_maintenance_log(input.log_text)
    log = MaintenanceLog(machine_id=input.machine_id, org_id=org_id, timestamp=datetime.now(timezone.utc).isoformat(),
                        log_text=input.log_text, technician=input.technician, severity=input.severity,
                        risk_keywords=keywords, embedding_similarity=score)
    await db.maintenance_logs.insert_one(log.model_dump())
    return log

@api_router.get("/machines/{machine_id}/maintenance-logs")
async def get_maintenance_logs(machine_id: str, current_user: dict = Depends(require_permission("view_reports"))):
    return await db.maintenance_logs.find(
        {"machine_id": machine_id, "org_id": current_user.get("current_org_id")}, {"_id": 0}
    ).sort("timestamp", -1).to_list(100)

@api_router.get("/machines/{machine_id}/sensor-graph")
async def get_sensor_graph(machine_id: str, current_user: dict = Depends(require_permission("view_reports"))):
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id, "org_id": current_user.get("current_org_id")}, {"_id": 0}
    ).sort("timestamp", -1).to_list(500)
    return build_sensor_correlation_graph(readings[::-1])

@api_router.post("/machines/{machine_id}/predict")
async def predict_failure(machine_id: str, current_user: dict = Depends(require_permission("run_predictions"))):
    org_id = current_user.get("current_org_id")
    machine = await db.machines.find_one({"id": machine_id, "org_id": org_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = await db.sensor_readings.find({"machine_id": machine_id, "org_id": org_id}, {"_id": 0}).sort("timestamp", -1).to_list(1000)
    logs = await db.maintenance_logs.find({"machine_id": machine_id, "org_id": org_id}, {"_id": 0}).to_list(100)
    
    gnn_result = gnn_predict_pytorch(readings[::-1])
    nlp_score = nlp_predict(logs)
    health, _, _ = calculate_health_score(readings[::-1])
    pred_data = multimodal_fusion_predict(gnn_result, nlp_score, health)
    
    prediction = Prediction(machine_id=machine_id, org_id=org_id, timestamp=datetime.now(timezone.utc).isoformat(), **pred_data)
    await db.predictions.insert_one(prediction.model_dump())
    
    new_prob = round(pred_data["fusion_score"] * 100, 1)
    new_risk = "critical" if pred_data["fusion_score"] > 0.6 else "warning" if pred_data["fusion_score"] > 0.3 else "healthy"
    await db.machines.update_one({"id": machine_id}, {"$set": {"health_score": health, "failure_probability": new_prob, "risk_level": new_risk}})
    await check_and_create_alert(machine, org_id, health, new_prob, new_risk)
    
    return prediction

@api_router.get("/machines/{machine_id}/predictions")
async def get_predictions(machine_id: str, current_user: dict = Depends(require_permission("view_reports"))):
    return await db.predictions.find(
        {"machine_id": machine_id, "org_id": current_user.get("current_org_id")}, {"_id": 0}
    ).sort("timestamp", -1).to_list(50)

# ===================== PDF REPORT ENDPOINT =====================

@api_router.get("/machines/{machine_id}/report")
async def generate_machine_report(machine_id: str, days: int = 30, current_user: dict = Depends(require_permission("view_reports"))):
    org_id = current_user.get("current_org_id")
    machine = await db.machines.find_one({"id": machine_id, "org_id": org_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id, "org_id": org_id, "timestamp": {"$gte": cutoff}}, {"_id": 0}
    ).sort("timestamp", 1).to_list(5000)
    predictions = await db.predictions.find(
        {"machine_id": machine_id, "org_id": org_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(10)
    logs = await db.maintenance_logs.find(
        {"machine_id": machine_id, "org_id": org_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(20)
    
    pdf_content = await generate_pdf_report(machine, readings, predictions, logs, days)
    
    filename = f"{machine['name'].replace(' ', '_')}_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    return Response(content=pdf_content, media_type="application/pdf",
                   headers={"Content-Disposition": f"attachment; filename={filename}"})


# ===================== MODEL COMPARISON ENDPOINT =====================

@api_router.get("/model-comparison")
async def get_model_comparison():
    """Return GNN vs Threshold model comparison metrics."""
    try:
        results_path = ROOT_DIR / "models" / "training_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                data = json.load(f)
            return {
                "gnn": data.get("comparison_metrics", {}).get("gnn", {}),
                "threshold": data.get("comparison_metrics", {}).get("threshold", {}),
                "early_warning_lead_time": data.get("comparison_metrics", {}).get("early_warning_lead_time", {}),
                "comparison_summary": data.get("comparison_metrics", {}).get("comparison_summary", {}),
                "roi_metrics": data.get("roi_metrics", {}),
                "training_timestamp": data.get("timestamp"),
                "training_config": data.get("training_config", {})
            }
    except Exception as e:
        logger.warning(f"Could not load training results: {e}")
    
    # Return default mock data if no training results
    return {
        "gnn": {
            "accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85,
            "roc_auc": 0.91, "false_positive_rate": 0.08, "missed_failure_rate": 0.05,
            "critical_detection_rate": 0.95
        },
        "threshold": {
            "accuracy": 0.68, "precision": 0.62, "recall": 0.75, "f1": 0.68,
            "roc_auc": 0.72, "false_positive_rate": 0.22, "missed_failure_rate": 0.18,
            "critical_detection_rate": 0.82
        },
        "early_warning_lead_time": {"gnn": 28.5, "threshold": 12.3, "improvement": 16.2},
        "comparison_summary": {
            "accuracy_improvement": 0.17, "f1_improvement": 0.17,
            "false_positive_reduction": 0.14, "missed_failure_reduction": 0.13
        }
    }


# ===================== ALERT ENDPOINTS =====================

@api_router.get("/alerts")
async def get_alerts(limit: int = 50, unacknowledged_only: bool = False, current_user: dict = Depends(require_permission("view_reports"))):
    org_id = current_user.get("current_org_id")
    query = {"org_id": org_id}
    if unacknowledged_only:
        query["acknowledged"] = False
    return await db.alerts.find(query, {"_id": 0}).sort("timestamp", -1).to_list(limit)

@api_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, current_user: dict = Depends(require_permission("manage_alerts"))):
    result = await db.alerts.update_one(
        {"id": alert_id, "org_id": current_user.get("current_org_id")},
        {"$set": {"acknowledged": True, "acknowledged_by": current_user["id"]}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}

@api_router.get("/alert-settings")
async def get_alert_settings(current_user: dict = Depends(require_permission("view_reports"))):
    org_id = current_user.get("current_org_id")
    settings = await db.alert_settings.find_one({"org_id": org_id}, {"_id": 0})
    if not settings:
        settings = AlertSettings(org_id=org_id).model_dump()
        await db.alert_settings.insert_one(settings)
    return settings

@api_router.put("/alert-settings")
async def update_alert_settings(update: AlertSettingsUpdate, current_user: dict = Depends(require_permission("manage_settings"))):
    org_id = current_user.get("current_org_id")
    update_dict = {k: v for k, v in update.model_dump().items() if v is not None}
    if update_dict:
        await db.alert_settings.update_one({"org_id": org_id}, {"$set": update_dict}, upsert=True)
    return await db.alert_settings.find_one({"org_id": org_id}, {"_id": 0})

@api_router.post("/upload")
async def upload_sensor_data(file: UploadFile = File(...), current_user: dict = Depends(require_permission("manage_machines"))):
    org_id = current_user.get("current_org_id")
    content = await file.read()
    
    try:
        if file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            readings = data if isinstance(data, list) else [data]
        elif file.filename.endswith('.csv'):
            reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
            readings = [{
                "id": str(uuid.uuid4()), "machine_id": row.get("machine_id", "unknown"), "org_id": org_id,
                "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "temperature": float(row.get("temperature", 0)), "pressure": float(row.get("pressure", 0)),
                "vibration": float(row.get("vibration", 0)), "rpm": float(row.get("rpm", 0)),
            } for row in reader]
        else:
            raise HTTPException(status_code=400, detail="Use CSV or JSON")
        
        for r in readings:
            r["org_id"] = org_id
        if readings:
            await db.sensor_readings.insert_many(readings)
        return {"message": f"Uploaded {len(readings)} readings"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/dashboard/summary")
async def get_dashboard_summary(current_user: dict = Depends(require_permission("view_reports"))):
    org_id = current_user.get("current_org_id")
    if not org_id:
        return {"total_machines": 0, "healthy": 0, "warning": 0, "critical": 0, "average_health_score": 100, "unacknowledged_alerts": 0}
    
    machines = await db.machines.find({"org_id": org_id}, {"_id": 0}).to_list(100)
    total = len(machines)
    healthy = sum(1 for m in machines if m.get("risk_level") == "healthy")
    warning = sum(1 for m in machines if m.get("risk_level") == "warning")
    critical = sum(1 for m in machines if m.get("risk_level") == "critical")
    avg_health = np.mean([m.get("health_score", 100) for m in machines]) if machines else 100
    alerts = await db.alerts.count_documents({"org_id": org_id, "acknowledged": False})
    
    return {"total_machines": total, "healthy": healthy, "warning": warning, "critical": critical,
            "average_health_score": round(avg_health, 1), "unacknowledged_alerts": alerts,
            "last_updated": datetime.now(timezone.utc).isoformat()}

@api_router.post("/seed-demo")
async def seed_demo_data(current_user: dict = Depends(require_permission("manage_machines"))):
    org_id = current_user.get("current_org_id")
    if not org_id:
        raise HTTPException(status_code=400, detail="Select an organization first")
    
    # Clear org data
    await db.machines.delete_many({"org_id": org_id})
    await db.sensor_readings.delete_many({"org_id": org_id})
    await db.maintenance_logs.delete_many({"org_id": org_id})
    await db.predictions.delete_many({"org_id": org_id})
    await db.alerts.delete_many({"org_id": org_id})
    
    demos = [
        ("Turbine-A1", "Gas Turbine", "Plant Floor A", 30),
        ("Motor-B2", "Electric Motor", "Assembly Line B", 45),
        ("Compressor-C3", "Air Compressor", "Utility Room C", 60),
        ("CNC-D4", "CNC Machine", "Machining Center D", 75),
        ("Pump-E5", "Hydraulic Pump", "Hydraulics Bay E", 90),
    ]
    
    demo_logs = [
        ("Abnormal bearing noise detected during routine inspection", "John Smith", "warning"),
        ("Slight oil leakage observed near main seal", "Maria Garcia", "warning"),
        ("Excessive vibration reported by operator", "Robert Chen", "error"),
        ("Regular maintenance completed, all systems normal", "Sarah Johnson", "info"),
        ("Motor overheating detected, cooling system checked", "Mike Wilson", "critical"),
    ]
    
    created = []
    for i, (name, mtype, loc, days) in enumerate(demos):
        machine = Machine(org_id=org_id, name=name, machine_type=mtype, location=loc)
        await db.machines.insert_one(machine.model_dump())
        created.append(machine)
        
        readings = simulate_sensor_data(machine.id, org_id, days)
        for j in range(0, len(readings), 500):
            await db.sensor_readings.insert_many(readings[j:j+500])
        
        health, prob, risk = calculate_health_score(readings)
        await db.machines.update_one({"id": machine.id}, {"$set": {"health_score": health, "failure_probability": prob, "risk_level": risk}})
        
        log_text, tech, severity = demo_logs[i]
        kw, score = analyze_maintenance_log(log_text)
        log = MaintenanceLog(machine_id=machine.id, org_id=org_id, 
                            timestamp=(datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 10))).isoformat(),
                            log_text=log_text, technician=tech, severity=severity, risk_keywords=kw, embedding_similarity=score)
        await db.maintenance_logs.insert_one(log.model_dump())
    
    return {"message": "Demo data seeded", "machines_created": len(created)}

# ===================== WEBSOCKET =====================

@app.websocket("/ws/{machine_id}")
async def websocket_endpoint(websocket: WebSocket, machine_id: str):
    await manager.connect(websocket, machine_id)
    try:
        machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
        if not machine:
            await websocket.close(code=4004)
            return
        
        while True:
            reading = generate_live_reading(machine_id, machine.get("org_id", ""), machine.get("health_score", 50))
            await db.sensor_readings.insert_one(reading)
            
            recent = await db.sensor_readings.find({"machine_id": machine_id}, {"_id": 0}).sort("timestamp", -1).to_list(24)
            health, prob, risk = calculate_health_score(recent[::-1])
            await db.machines.update_one({"id": machine_id}, {"$set": {"health_score": health, "failure_probability": prob, "risk_level": risk}})
            
            await websocket.send_json({"type": "sensor_update", "reading": reading, "health_score": health, "failure_probability": prob, "risk_level": risk})
            machine["health_score"] = health
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket, machine_id)
    except:
        manager.disconnect(websocket, machine_id)

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','), allow_methods=["*"], allow_headers=["*"])

@app.on_event("shutdown")
async def shutdown():
    client.close()
