from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import bcrypt
import jwt
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'predictmaint-secret-key-change-in-production')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Create the main app
app = FastAPI(title="Multimodal Predictive Maintenance API")
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SendGrid Config
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'alerts@predictmaint.com')

# ===================== AUTH MODELS =====================

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
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str

# ===================== AUTH FUNCTIONS =====================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
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
    
    return user

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
        if machine_id in self.active_connections:
            if websocket in self.active_connections[machine_id]:
                self.active_connections[machine_id].remove(websocket)
    
    async def broadcast(self, machine_id: str, data: dict):
        if machine_id in self.active_connections:
            for connection in self.active_connections[machine_id]:
                try:
                    await connection.send_json(data)
                except:
                    pass
    
    async def broadcast_all(self, data: dict):
        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    await connection.send_json(data)
                except:
                    pass

manager = ConnectionManager()

# ===================== GNN MODELS =====================

class SensorGCN(nn.Module):
    def __init__(self, num_features: int = 4, hidden_channels: int = 32, num_classes: int = 3):
        super(SensorGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        x = self.lin(x)
        return F.softmax(x, dim=1)

class SensorGAT(nn.Module):
    def __init__(self, num_features: int = 4, hidden_channels: int = 32, num_heads: int = 4, num_classes: int = 3):
        super(SensorGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.3)
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        x = self.lin(x)
        return F.softmax(x, dim=1)

# Initialize models
gcn_model = SensorGCN(num_features=4, hidden_channels=32, num_classes=3)
gat_model = SensorGAT(num_features=4, hidden_channels=32, num_heads=4, num_classes=3)

# Try to load trained weights
MODEL_DIR = ROOT_DIR / "models"
if (MODEL_DIR / "gcn_cmapss.pt").exists():
    try:
        gcn_model.load_state_dict(torch.load(MODEL_DIR / "gcn_cmapss.pt", map_location='cpu'))
        logger.info("Loaded trained GCN weights")
    except:
        logger.warning("Could not load GCN weights, using random initialization")

if (MODEL_DIR / "gat_cmapss.pt").exists():
    try:
        gat_model.load_state_dict(torch.load(MODEL_DIR / "gat_cmapss.pt", map_location='cpu'))
        logger.info("Loaded trained GAT weights")
    except:
        logger.warning("Could not load GAT weights, using random initialization")

gcn_model.eval()
gat_model.eval()

# ===================== PYDANTIC MODELS =====================

class Machine(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str  # Multi-tenant: owner
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
    user_id: str
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
    user_id: str
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
    user_id: str
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
    user_id: str
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
    user_id: str
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
    if not SENDGRID_API_KEY or not recipients:
        return False
    
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        severity_color = {"critical": "#ef4444", "warning": "#facc15", "info": "#3b82f6"}.get(alert.severity, "#3b82f6")
        
        html_content = f"""
        <html><body style="font-family: Arial, sans-serif; background-color: #1a1a1a; color: #e4e4e7; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: #27272a; border-radius: 8px; padding: 24px;">
                <h1 style="color: {severity_color};">⚠️ {alert.alert_type.upper()} ALERT</h1>
                <div style="background-color: #18181b; border-left: 4px solid {severity_color}; padding: 16px; margin: 16px 0;">
                    <h2 style="margin: 0 0 8px 0; color: #e4e4e7;">{alert.machine_name}</h2>
                    <p style="margin: 0; color: #a1a1aa;">{alert.message}</p>
                </div>
                <p>Health Score: <strong style="color: {severity_color};">{alert.health_score:.1f}%</strong></p>
                <p>Failure Probability: <strong>{alert.failure_probability:.1f}%</strong></p>
            </div>
        </body></html>
        """
        
        for recipient in recipients:
            message = Mail(
                from_email=Email(SENDER_EMAIL, "PredictMaint Alerts"),
                to_emails=To(recipient),
                subject=f"[{alert.severity.upper()}] {alert.machine_name} - {alert.alert_type}",
                html_content=Content("text/html", html_content)
            )
            sg.send(message)
        
        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False

# ===================== DATA SIMULATION =====================

def generate_degradation_pattern(days: int) -> np.ndarray:
    x = np.linspace(0, 1, days)
    degradation = 1 - np.exp(-3 * (x ** 2))
    noise = np.random.normal(0, 0.02, days)
    return np.clip(degradation + noise, 0, 1)

def simulate_sensor_data(machine_id: str, user_id: str, days: int = 90) -> List[dict]:
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
        
        readings.append({
            "id": str(uuid.uuid4()),
            "machine_id": machine_id,
            "user_id": user_id,
            "timestamp": (start_time + timedelta(hours=i)).isoformat(),
            "temperature": round(float(temp), 2),
            "pressure": round(float(pressure), 2),
            "vibration": round(float(vibration), 3),
            "rpm": round(float(rpm), 1),
            "voltage": round(float(220 + np.random.normal(0, 5)), 1),
            "current": round(float(15 + d * 10 + np.random.normal(0, 1)), 2)
        })
    
    return readings

def generate_live_reading(machine_id: str, user_id: str, base_health: float) -> dict:
    degradation = 1 - (base_health / 100)
    return {
        "id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": round(float(45 + degradation * 35 + np.random.normal(0, 2)), 2),
        "pressure": round(float(100 - degradation * 25 + np.random.normal(0, 3)), 2),
        "vibration": round(float(0.5 + degradation * 4.5 + np.random.normal(0, 0.1)), 3),
        "rpm": round(float(3000 - degradation * 500 + np.random.normal(0, 50)), 1),
        "voltage": round(float(220 + np.random.normal(0, 5)), 1),
        "current": round(float(15 + degradation * 10 + np.random.normal(0, 1)), 2)
    }

def calculate_health_score(readings: List[dict]) -> tuple:
    if not readings:
        return 100.0, 0.0, "healthy"
    
    recent = readings[-24:]
    temps = [r["temperature"] for r in recent]
    vibs = [r["vibration"] for r in recent]
    pressures = [r["pressure"] for r in recent]
    
    temp_score = max(0, 100 - (np.mean(temps) - 45) * 2)
    vib_score = max(0, 100 - np.mean(vibs) * 20)
    pressure_score = max(0, min(100, np.mean(pressures)))
    
    health_score = max(0, min(100, temp_score * 0.3 + vib_score * 0.4 + pressure_score * 0.3))
    failure_prob = max(0, min(100, (100 - health_score) * 1.2))
    
    risk_level = "critical" if health_score < 40 else "warning" if health_score < 70 else "healthy"
    
    return round(health_score, 1), round(failure_prob, 1), risk_level

# ===================== GNN FUNCTIONS =====================

def build_pyg_graph(readings: List[dict]) -> Data:
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    
    if len(readings) < 10:
        x = torch.tensor([[50.0, 100.0, 0.5, 3000.0]] * 4, dtype=torch.float)
        edge_index = torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=torch.ones(6))
    
    data = {s: [r[s] for r in readings[-500:] if s in r] for s in sensors}
    
    node_features = []
    for s in sensors:
        if data[s]:
            node_features.append([np.mean(data[s]), np.std(data[s]), np.min(data[s]), np.max(data[s])])
        else:
            node_features.append([0.0, 0.0, 0.0, 0.0])
    
    x = torch.tensor(node_features, dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
    
    edges_src, edges_dst, edge_weights = [], [], []
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) and len(data[s1]) > 2:
                try:
                    corr, _ = stats.pearsonr(data[s1], data[s2])
                    if abs(corr) > 0.2:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                        edge_weights.extend([abs(corr), abs(corr)])
                except:
                    pass
    
    if not edges_src:
        for i in range(4):
            for j in range(i+1, 4):
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                edge_weights.extend([0.5, 0.5])
    
    return Data(x=x, edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long), 
                edge_attr=torch.tensor(edge_weights, dtype=torch.float))

def build_sensor_correlation_graph(readings: List[dict]) -> Dict:
    if len(readings) < 10:
        return {
            "nodes": [{"id": s, "group": i % 3 + 1, "value": 50} for i, s in enumerate(["temperature", "pressure", "vibration", "rpm"])],
            "links": [{"source": "temperature", "target": "vibration", "weight": 0.5}]
        }
    
    sensors = ["temperature", "pressure", "vibration", "rpm"]
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
        gcn_out = gcn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        gcn_probs = gcn_out.squeeze().tolist()
        
        gat_out = gat_model(graph_data.x, graph_data.edge_index)
        gat_probs = gat_out.squeeze().tolist()
    
    ensemble_probs = [(g + a) / 2 for g, a in zip(gcn_probs, gat_probs)]
    risk_score = ensemble_probs[1] * 0.3 + ensemble_probs[2] * 0.7
    
    return {
        "gcn_prediction": gcn_probs,
        "gat_prediction": gat_probs,
        "ensemble_prediction": ensemble_probs,
        "risk_score": float(risk_score),
        "predicted_class": ["healthy", "warning", "critical"][np.argmax(ensemble_probs)]
    }

# ===================== NLP FUNCTIONS =====================

RISK_KEYWORDS = ["abnormal", "noise", "leak", "leakage", "vibration", "excessive", "overheating", 
                 "failure", "broken", "crack", "worn", "degraded", "malfunction", "error", 
                 "warning", "critical", "urgent", "bearing", "motor", "seal", "belt", "corrosion"]

def analyze_maintenance_log(text: str) -> tuple:
    text_lower = text.lower()
    found_keywords = [kw for kw in RISK_KEYWORDS if kw in text_lower]
    return found_keywords, min(1.0, len(found_keywords) * 0.15)

def nlp_predict(logs: List[dict]) -> float:
    if not logs:
        return 0.0
    
    total_risk = sum(
        len(log.get("risk_keywords", [])) * 0.1 + 
        {"info": 0, "warning": 0.2, "error": 0.4, "critical": 0.6}.get(log.get("severity", "info"), 0)
        for log in logs[-10:]
    )
    return min(1.0, total_risk / min(len(logs), 10))

def multimodal_fusion_predict(gnn_result: Dict, nlp_score: float, health_score: float) -> dict:
    gnn_score = gnn_result["risk_score"]
    fusion_score = min(1.0, max(0.0, 0.5 * gnn_score + 0.25 * nlp_score + 0.25 * (1 - health_score / 100)))
    
    if fusion_score < 0.2:
        rul_days, failure_type = 90, "None predicted"
    elif fusion_score < 0.4:
        rul_days, failure_type = 60, "Minor wear"
    elif fusion_score < 0.6:
        rul_days, failure_type = 30, "Component degradation"
    elif fusion_score < 0.8:
        rul_days, failure_type = 14, "Bearing failure likely"
    else:
        rul_days, failure_type = 7, "Imminent failure"
    
    rul_days += np.random.uniform(-5, 5)
    
    return {
        "predicted_failure_date": (datetime.now(timezone.utc) + timedelta(days=rul_days)).isoformat(),
        "remaining_useful_life_days": round(rul_days, 1),
        "confidence_score": round(0.6 + 0.3 * (1 - abs(fusion_score - 0.5) * 2), 2),
        "failure_type": failure_type,
        "gnn_score": round(gnn_score, 3),
        "nlp_score": round(nlp_score, 3),
        "fusion_score": round(fusion_score, 3),
        "gcn_prediction": gnn_result["gcn_prediction"],
        "gat_prediction": gnn_result["gat_prediction"]
    }

# ===================== ALERT SYSTEM =====================

async def check_and_create_alert(machine: dict, user_id: str, health_score: float, failure_prob: float, risk_level: str):
    settings = await db.alert_settings.find_one({"user_id": user_id}, {"_id": 0})
    if not settings:
        settings = {"email_enabled": True, "email_recipients": [], "critical_threshold": 40.0, "warning_threshold": 70.0}
    
    alert_type, severity, message = None, None, None
    
    if health_score < settings["critical_threshold"]:
        alert_type, severity = "Critical Health", "critical"
        message = f"Machine health score dropped to {health_score:.1f}%. Immediate attention required!"
    elif health_score < settings["warning_threshold"]:
        alert_type, severity = "Health Warning", "warning"
        message = f"Machine health score at {health_score:.1f}%. Schedule maintenance soon."
    
    if alert_type:
        alert = Alert(
            user_id=user_id,
            machine_id=machine["id"],
            machine_name=machine.get("name", "Unknown"),
            alert_type=alert_type,
            severity=severity,
            message=message,
            health_score=health_score,
            failure_probability=failure_prob,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        await db.alerts.insert_one(alert.model_dump())
        await manager.broadcast_all({"type": "alert", "data": alert.model_dump()})
        
        if settings["email_enabled"] and settings["email_recipients"]:
            email_sent = await send_alert_email(alert, settings["email_recipients"])
            if email_sent:
                await db.alerts.update_one({"id": alert.id}, {"$set": {"email_sent": True}})
        
        return alert
    return None

# ===================== AUTH ENDPOINTS =====================

@api_router.post("/auth/register")
async def register(input: UserCreate):
    existing = await db.users.find_one({"email": input.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=input.email,
        name=input.name,
        password_hash=hash_password(input.password)
    )
    
    await db.users.insert_one(user.model_dump())
    token = create_access_token(user.id, user.email)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at}
    }

@api_router.post("/auth/login")
async def login(input: UserLogin):
    user = await db.users.find_one({"email": input.email})
    if not user or not verify_password(input.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token(user["id"], user["email"])
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "name": user["name"], "created_at": user["created_at"]}
    }

@api_router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

# ===================== API ENDPOINTS =====================

@api_router.get("/")
async def root():
    return {"message": "Multimodal Predictive Maintenance API", "status": "operational", "version": "3.0"}

@api_router.post("/machines", response_model=Machine)
async def create_machine(input: MachineCreate, current_user: dict = Depends(get_current_user)):
    machine = Machine(user_id=current_user["id"], **input.model_dump())
    await db.machines.insert_one(machine.model_dump())
    return machine

@api_router.get("/machines")
async def get_machines(current_user: dict = Depends(get_current_user)):
    machines = await db.machines.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(100)
    return machines

@api_router.get("/machines/{machine_id}")
async def get_machine(machine_id: str, current_user: dict = Depends(get_current_user)):
    machine = await db.machines.find_one({"id": machine_id, "user_id": current_user["id"]}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@api_router.delete("/machines/{machine_id}")
async def delete_machine(machine_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.machines.delete_one({"id": machine_id, "user_id": current_user["id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Machine not found")
    await db.sensor_readings.delete_many({"machine_id": machine_id, "user_id": current_user["id"]})
    await db.maintenance_logs.delete_many({"machine_id": machine_id, "user_id": current_user["id"]})
    await db.predictions.delete_many({"machine_id": machine_id, "user_id": current_user["id"]})
    return {"message": "Machine deleted"}

@api_router.get("/machines/{machine_id}/readings")
async def get_sensor_readings(machine_id: str, limit: int = 500, current_user: dict = Depends(get_current_user)):
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id, "user_id": current_user["id"]}, {"_id": 0}
    ).sort("timestamp", -1).to_list(limit)
    return readings[::-1]

@api_router.post("/machines/{machine_id}/simulate")
async def simulate_machine_data(machine_id: str, days: int = 90, current_user: dict = Depends(get_current_user)):
    machine = await db.machines.find_one({"id": machine_id, "user_id": current_user["id"]}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = simulate_sensor_data(machine_id, current_user["id"], days)
    
    for i in range(0, len(readings), 500):
        await db.sensor_readings.insert_many(readings[i:i+500])
    
    health_score, failure_prob, risk_level = calculate_health_score(readings)
    await db.machines.update_one(
        {"id": machine_id}, 
        {"$set": {"health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}}
    )
    
    await check_and_create_alert(machine, current_user["id"], health_score, failure_prob, risk_level)
    
    return {"message": f"Simulated {len(readings)} readings", "health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}

@api_router.post("/maintenance-logs")
async def create_maintenance_log(input: MaintenanceLogCreate, current_user: dict = Depends(get_current_user)):
    risk_keywords, risk_score = analyze_maintenance_log(input.log_text)
    
    log = MaintenanceLog(
        machine_id=input.machine_id,
        user_id=current_user["id"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        log_text=input.log_text,
        technician=input.technician,
        severity=input.severity,
        risk_keywords=risk_keywords,
        embedding_similarity=risk_score
    )
    
    await db.maintenance_logs.insert_one(log.model_dump())
    return log

@api_router.get("/machines/{machine_id}/maintenance-logs")
async def get_maintenance_logs(machine_id: str, current_user: dict = Depends(get_current_user)):
    logs = await db.maintenance_logs.find(
        {"machine_id": machine_id, "user_id": current_user["id"]}, {"_id": 0}
    ).sort("timestamp", -1).to_list(100)
    return logs

@api_router.get("/machines/{machine_id}/sensor-graph")
async def get_sensor_graph(machine_id: str, current_user: dict = Depends(get_current_user)):
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id, "user_id": current_user["id"]}, {"_id": 0}
    ).sort("timestamp", -1).to_list(500)
    return build_sensor_correlation_graph(readings[::-1])

@api_router.post("/machines/{machine_id}/predict")
async def predict_failure(machine_id: str, current_user: dict = Depends(get_current_user)):
    machine = await db.machines.find_one({"id": machine_id, "user_id": current_user["id"]}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id, "user_id": current_user["id"]}, {"_id": 0}
    ).sort("timestamp", -1).to_list(1000)
    readings = readings[::-1]
    
    logs = await db.maintenance_logs.find(
        {"machine_id": machine_id, "user_id": current_user["id"]}, {"_id": 0}
    ).to_list(100)
    
    gnn_result = gnn_predict_pytorch(readings)
    nlp_score = nlp_predict(logs)
    health_score, _, _ = calculate_health_score(readings)
    prediction_data = multimodal_fusion_predict(gnn_result, nlp_score, health_score)
    
    prediction = Prediction(machine_id=machine_id, user_id=current_user["id"], 
                           timestamp=datetime.now(timezone.utc).isoformat(), **prediction_data)
    await db.predictions.insert_one(prediction.model_dump())
    
    new_failure_prob = round(prediction_data["fusion_score"] * 100, 1)
    new_risk_level = "critical" if prediction_data["fusion_score"] > 0.6 else "warning" if prediction_data["fusion_score"] > 0.3 else "healthy"
    
    await db.machines.update_one(
        {"id": machine_id},
        {"$set": {"health_score": health_score, "failure_probability": new_failure_prob, "risk_level": new_risk_level}}
    )
    
    await check_and_create_alert(machine, current_user["id"], health_score, new_failure_prob, new_risk_level)
    
    return prediction

@api_router.get("/machines/{machine_id}/predictions")
async def get_predictions(machine_id: str, current_user: dict = Depends(get_current_user)):
    predictions = await db.predictions.find(
        {"machine_id": machine_id, "user_id": current_user["id"]}, {"_id": 0}
    ).sort("timestamp", -1).to_list(50)
    return predictions

@api_router.get("/alerts")
async def get_alerts(limit: int = 50, unacknowledged_only: bool = False, current_user: dict = Depends(get_current_user)):
    query = {"user_id": current_user["id"]}
    if unacknowledged_only:
        query["acknowledged"] = False
    alerts = await db.alerts.find(query, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return alerts

@api_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.alerts.update_one(
        {"id": alert_id, "user_id": current_user["id"]},
        {"$set": {"acknowledged": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}

@api_router.get("/alert-settings")
async def get_alert_settings(current_user: dict = Depends(get_current_user)):
    settings = await db.alert_settings.find_one({"user_id": current_user["id"]}, {"_id": 0})
    if not settings:
        settings = AlertSettings(user_id=current_user["id"]).model_dump()
        await db.alert_settings.insert_one(settings)
    return settings

@api_router.put("/alert-settings")
async def update_alert_settings(update: AlertSettingsUpdate, current_user: dict = Depends(get_current_user)):
    update_dict = {k: v for k, v in update.model_dump().items() if v is not None}
    if update_dict:
        await db.alert_settings.update_one(
            {"user_id": current_user["id"]},
            {"$set": update_dict},
            upsert=True
        )
    return await db.alert_settings.find_one({"user_id": current_user["id"]}, {"_id": 0})

@api_router.post("/upload")
async def upload_sensor_data(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    content = await file.read()
    
    try:
        if file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            readings = data if isinstance(data, list) else [data]
        elif file.filename.endswith('.csv'):
            reader = csv.DictReader(io.StringIO(content.decode('utf-8')))
            readings = [{
                "id": str(uuid.uuid4()),
                "machine_id": row.get("machine_id", "unknown"),
                "user_id": current_user["id"],
                "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "temperature": float(row.get("temperature", row.get("temp", 0))),
                "pressure": float(row.get("pressure", 0)),
                "vibration": float(row.get("vibration", 0)),
                "rpm": float(row.get("rpm", 0)),
                "voltage": float(row.get("voltage", 0)) if row.get("voltage") else None,
                "current": float(row.get("current", 0)) if row.get("current") else None
            } for row in reader]
        else:
            raise HTTPException(status_code=400, detail="Use CSV or JSON format")
        
        for r in readings:
            r["user_id"] = current_user["id"]
        
        if readings:
            await db.sensor_readings.insert_many(readings)
        
        return {"message": f"Uploaded {len(readings)} readings", "count": len(readings)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/dashboard/summary")
async def get_dashboard_summary(current_user: dict = Depends(get_current_user)):
    machines = await db.machines.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(100)
    
    total_machines = len(machines)
    healthy = sum(1 for m in machines if m.get("risk_level") == "healthy")
    warning = sum(1 for m in machines if m.get("risk_level") == "warning")
    critical = sum(1 for m in machines if m.get("risk_level") == "critical")
    avg_health = np.mean([m.get("health_score", 100) for m in machines]) if machines else 100
    alert_count = await db.alerts.count_documents({"user_id": current_user["id"], "acknowledged": False})
    
    return {
        "total_machines": total_machines,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "average_health_score": round(avg_health, 1),
        "unacknowledged_alerts": alert_count,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

@api_router.post("/seed-demo")
async def seed_demo_data(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]
    
    # Clear user's data
    await db.machines.delete_many({"user_id": user_id})
    await db.sensor_readings.delete_many({"user_id": user_id})
    await db.maintenance_logs.delete_many({"user_id": user_id})
    await db.predictions.delete_many({"user_id": user_id})
    await db.alerts.delete_many({"user_id": user_id})
    
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
        machine = Machine(user_id=user_id, **m_data)
        await db.machines.insert_one(machine.model_dump())
        created_machines.append(machine)
        
        days = 30 + i * 15
        readings = simulate_sensor_data(machine.id, user_id, days)
        
        for j in range(0, len(readings), 500):
            await db.sensor_readings.insert_many(readings[j:j+500])
        
        health_score, failure_prob, risk_level = calculate_health_score(readings)
        await db.machines.update_one(
            {"id": machine.id},
            {"$set": {"health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}}
        )
        
        log_data = demo_logs[i % len(demo_logs)]
        risk_keywords, risk_score = analyze_maintenance_log(log_data["log_text"])
        log = MaintenanceLog(
            machine_id=machine.id,
            user_id=user_id,
            timestamp=(datetime.now(timezone.utc) - timedelta(days=np.random.randint(1, 10))).isoformat(),
            log_text=log_data["log_text"],
            technician=log_data["technician"],
            severity=log_data["severity"],
            risk_keywords=risk_keywords,
            embedding_similarity=risk_score
        )
        await db.maintenance_logs.insert_one(log.model_dump())
    
    return {"message": "Demo data seeded", "machines_created": len(created_machines), "machine_ids": [m.id for m in created_machines]}

# WebSocket endpoint
@app.websocket("/ws/{machine_id}")
async def websocket_endpoint(websocket: WebSocket, machine_id: str):
    await manager.connect(websocket, machine_id)
    
    try:
        machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
        if not machine:
            await websocket.close(code=4004)
            return
        
        while True:
            reading = generate_live_reading(machine_id, machine.get("user_id", ""), machine.get("health_score", 50))
            await db.sensor_readings.insert_one(reading)
            
            recent_readings = await db.sensor_readings.find(
                {"machine_id": machine_id}, {"_id": 0}
            ).sort("timestamp", -1).to_list(24)
            
            health_score, failure_prob, risk_level = calculate_health_score(recent_readings[::-1])
            
            await db.machines.update_one(
                {"id": machine_id},
                {"$set": {"health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}}
            )
            
            await websocket.send_json({
                "type": "sensor_update",
                "reading": reading,
                "health_score": health_score,
                "failure_probability": failure_prob,
                "risk_level": risk_level
            })
            
            machine["health_score"] = health_score
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, machine_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, machine_id)

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
