from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, status
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
from passlib.context import CryptContext
from jose import JWTError, jwt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Multimodal Predictive Maintenance API")
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== JWT & AUTH CONFIG =====================
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'predictmaint-secret-key-change-in-production-2024')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# ===================== SENDGRID CONFIG =====================
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'alerts@predictmaint.com')

# ===================== AUTH UTILITIES =====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        user = await db.users.find_one({"id": user_id}, {"_id": 0, "password_hash": 0})
        return user
    except JWTError:
        return None

async def get_required_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
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

# ===================== GNN MODELS (Trained on CMAPSS) =====================

class SensorGCN(nn.Module):
    def __init__(self, num_features: int = 1, hidden_channels: int = 64, num_classes: int = 3):
        super(SensorGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.bn3(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return F.softmax(self.lin2(x), dim=1)

class SensorGAT(nn.Module):
    def __init__(self, num_features: int = 1, hidden_channels: int = 32, num_heads: int = 4, num_classes: int = 3):
        super(SensorGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.3)
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        return F.softmax(self.lin(x), dim=1)

# Load trained models
gcn_model = SensorGCN(num_features=1, hidden_channels=64, num_classes=3)
gat_model = SensorGAT(num_features=1, hidden_channels=32, num_heads=4, num_classes=3)

MODEL_DIR = ROOT_DIR / "models"
if (MODEL_DIR / "gcn_cmapss.pt").exists():
    try:
        gcn_model.load_state_dict(torch.load(MODEL_DIR / "gcn_cmapss.pt", map_location='cpu'))
        logger.info("Loaded trained GCN model")
    except Exception as e:
        logger.warning(f"Could not load GCN model: {e}")

if (MODEL_DIR / "gat_cmapss.pt").exists():
    try:
        gat_model.load_state_dict(torch.load(MODEL_DIR / "gat_cmapss.pt", map_location='cpu'))
        logger.info("Loaded trained GAT model")
    except Exception as e:
        logger.warning(f"Could not load GAT model: {e}")

gcn_model.eval()
gat_model.eval()

# ===================== PYDANTIC MODELS =====================

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User

class Machine(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
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
    user_id: str = ""
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
    user_id: str = ""
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
    user_id: str = ""
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
    user_id: str = ""
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
    user_id: str = ""
    email_enabled: bool = True
    email_recipients: List[str] = []
    critical_threshold: float = 40.0
    warning_threshold: float = 70.0

class AlertSettingsUpdate(BaseModel):
    email_enabled: Optional[bool] = None
    email_recipients: Optional[List[str]] = None
    critical_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None

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
        
        reading = {
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
        }
        readings.append(reading)
    
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

# ===================== GNN PREDICTION =====================

def build_sensor_graph_tensor(readings: List[dict]) -> Data:
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    
    if len(readings) < 10:
        x = torch.randn(4, 1)
        edge_index = torch.tensor([[i, j] for i in range(4) for j in range(4) if i != j], dtype=torch.long).t()
        return Data(x=x, edge_index=edge_index)
    
    # Extract sensor statistics
    data = {s: [r[s] for r in readings[-100:] if s in r] for s in sensors}
    
    node_features = []
    for s in sensors:
        if data[s]:
            node_features.append([np.mean(data[s])])
        else:
            node_features.append([0.0])
    
    x = torch.tensor(node_features, dtype=torch.float)
    x = (x - x.mean()) / (x.std() + 1e-8)
    
    edge_index = torch.tensor([[i, j] for i in range(4) for j in range(4) if i != j], dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index)

def gnn_predict_pytorch(readings: List[dict]) -> Dict:
    graph_data = build_sensor_graph_tensor(readings)
    
    with torch.no_grad():
        gcn_out = gcn_model(graph_data.x, graph_data.edge_index)
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

def build_sensor_correlation_graph(readings: List[dict]) -> Dict:
    if len(readings) < 10:
        return {
            "nodes": [{"id": s, "group": i % 3 + 1, "value": 50} for i, s in enumerate(["temperature", "pressure", "vibration", "rpm"])],
            "links": [{"source": "temperature", "target": "vibration", "weight": 0.5}]
        }
    
    sensors = ["temperature", "pressure", "vibration", "rpm"]
    data = {s: [r[s] for r in readings[-500:] if s in r] for s in sensors}
    
    nodes = [{"id": s, "group": (i % 3) + 1, "value": min(100, np.var(data[s]) * 10) if data[s] else 50} for i, s in enumerate(sensors)]
    
    links = []
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) and len(data[s1]) > 2:
                try:
                    corr, _ = stats.pearsonr(data[s1], data[s2])
                    if abs(corr) > 0.3:
                        links.append({"source": s1, "target": s2, "weight": round(abs(corr), 3)})
                except:
                    pass
    
    return {"nodes": nodes, "links": links}

# ===================== NLP PROCESSING =====================

RISK_KEYWORDS = ["abnormal", "noise", "leak", "leakage", "vibration", "excessive", "overheating", "failure", "broken", "crack", "worn", "degraded", "malfunction", "error", "warning", "critical", "urgent", "bearing", "motor", "seal", "belt", "corrosion", "fatigue", "alignment"]

def analyze_maintenance_log(text: str) -> tuple:
    text_lower = text.lower()
    found_keywords = [kw for kw in RISK_KEYWORDS if kw in text_lower]
    risk_score = min(1.0, len(found_keywords) * 0.15)
    return found_keywords, risk_score

def nlp_predict(logs: List[dict]) -> float:
    if not logs:
        return 0.0
    
    recent_logs = logs[-10:]
    total_risk = sum(
        len(log.get("risk_keywords", [])) * 0.1 + {"info": 0, "warning": 0.2, "error": 0.4, "critical": 0.6}.get(log.get("severity", "info"), 0)
        for log in recent_logs
    )
    return min(1.0, total_risk / len(recent_logs))

# ===================== MULTIMODAL FUSION =====================

def multimodal_fusion_predict(gnn_result: Dict, nlp_score: float, health_score: float) -> dict:
    gnn_score = gnn_result["risk_score"]
    fusion_score = min(1.0, max(0.0, 0.5 * gnn_score + 0.25 * nlp_score + 0.25 * (1 - health_score / 100)))
    
    if fusion_score < 0.2:
        rul_days, failure_type = 90 + np.random.uniform(-10, 10), "None predicted"
    elif fusion_score < 0.4:
        rul_days, failure_type = 60 + np.random.uniform(-10, 10), "Minor wear"
    elif fusion_score < 0.6:
        rul_days, failure_type = 30 + np.random.uniform(-5, 5), "Component degradation"
    elif fusion_score < 0.8:
        rul_days, failure_type = 14 + np.random.uniform(-3, 3), "Bearing failure likely"
    else:
        rul_days, failure_type = 7 + np.random.uniform(-2, 2), "Imminent failure"
    
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
            machine_id=machine["id"],
            user_id=user_id,
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
        return alert
    return None

# ===================== AUTH ENDPOINTS =====================

@api_router.post("/auth/register", response_model=Token)
async def register(input: UserCreate):
    existing = await db.users.find_one({"email": input.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(email=input.email, name=input.name)
    user_doc = user.model_dump()
    user_doc["password_hash"] = get_password_hash(input.password)
    
    await db.users.insert_one(user_doc)
    
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    return Token(access_token=access_token, user=user)

@api_router.post("/auth/login", response_model=Token)
async def login(input: UserLogin):
    user_doc = await db.users.find_one({"email": input.email})
    if not user_doc or not verify_password(input.password, user_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user = User(id=user_doc["id"], email=user_doc["email"], name=user_doc["name"], created_at=user_doc.get("created_at", ""))
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    return Token(access_token=access_token, user=user)

@api_router.get("/auth/me")
async def get_me(user: dict = Depends(get_required_user)):
    return user

# ===================== API ENDPOINTS =====================

@api_router.get("/")
async def root():
    return {"message": "Multimodal Predictive Maintenance API", "status": "operational", "version": "3.0", "auth": "JWT"}

# Machine endpoints (multi-tenant)
@api_router.post("/machines", response_model=Machine)
async def create_machine(input: MachineCreate, user: dict = Depends(get_required_user)):
    machine = Machine(**input.model_dump(), user_id=user["id"])
    await db.machines.insert_one(machine.model_dump())
    return machine

@api_router.get("/machines", response_model=List[Machine])
async def get_machines(user: dict = Depends(get_required_user)):
    machines = await db.machines.find({"user_id": user["id"]}, {"_id": 0}).to_list(100)
    return machines

@api_router.get("/machines/{machine_id}", response_model=Machine)
async def get_machine(machine_id: str, user: dict = Depends(get_required_user)):
    machine = await db.machines.find_one({"id": machine_id, "user_id": user["id"]}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@api_router.delete("/machines/{machine_id}")
async def delete_machine(machine_id: str, user: dict = Depends(get_required_user)):
    result = await db.machines.delete_one({"id": machine_id, "user_id": user["id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Machine not found")
    await db.sensor_readings.delete_many({"machine_id": machine_id, "user_id": user["id"]})
    await db.maintenance_logs.delete_many({"machine_id": machine_id, "user_id": user["id"]})
    await db.predictions.delete_many({"machine_id": machine_id, "user_id": user["id"]})
    await db.alerts.delete_many({"machine_id": machine_id, "user_id": user["id"]})
    return {"message": "Machine deleted"}

# Sensor reading endpoints
@api_router.get("/machines/{machine_id}/readings")
async def get_sensor_readings(machine_id: str, limit: int = 500, user: dict = Depends(get_required_user)):
    readings = await db.sensor_readings.find({"machine_id": machine_id, "user_id": user["id"]}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return readings[::-1]

@api_router.post("/machines/{machine_id}/simulate")
async def simulate_machine_data(machine_id: str, days: int = 90, user: dict = Depends(get_required_user)):
    machine = await db.machines.find_one({"id": machine_id, "user_id": user["id"]}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = simulate_sensor_data(machine_id, user["id"], days)
    
    for i in range(0, len(readings), 500):
        await db.sensor_readings.insert_many(readings[i:i+500])
    
    health_score, failure_prob, risk_level = calculate_health_score(readings)
    await db.machines.update_one({"id": machine_id}, {"$set": {"health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}})
    
    await check_and_create_alert(machine, user["id"], health_score, failure_prob, risk_level)
    
    return {"message": f"Simulated {len(readings)} sensor readings", "health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}

# Maintenance log endpoints
@api_router.post("/maintenance-logs", response_model=MaintenanceLog)
async def create_maintenance_log(input: MaintenanceLogCreate, user: dict = Depends(get_required_user)):
    risk_keywords, risk_score = analyze_maintenance_log(input.log_text)
    log = MaintenanceLog(
        machine_id=input.machine_id,
        user_id=user["id"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        log_text=input.log_text,
        technician=input.technician,
        severity=input.severity,
        risk_keywords=risk_keywords,
        embedding_similarity=risk_score
    )
    await db.maintenance_logs.insert_one(log.model_dump())
    return log

@api_router.get("/machines/{machine_id}/maintenance-logs", response_model=List[MaintenanceLog])
async def get_maintenance_logs(machine_id: str, user: dict = Depends(get_required_user)):
    logs = await db.maintenance_logs.find({"machine_id": machine_id, "user_id": user["id"]}, {"_id": 0}).sort("timestamp", -1).to_list(100)
    return logs

# Graph endpoints
@api_router.get("/machines/{machine_id}/sensor-graph")
async def get_sensor_graph(machine_id: str, user: dict = Depends(get_required_user)):
    readings = await db.sensor_readings.find({"machine_id": machine_id, "user_id": user["id"]}, {"_id": 0}).sort("timestamp", -1).to_list(500)
    return build_sensor_correlation_graph(readings[::-1])

# Prediction endpoints
@api_router.post("/machines/{machine_id}/predict")
async def predict_failure(machine_id: str, user: dict = Depends(get_required_user)):
    machine = await db.machines.find_one({"id": machine_id, "user_id": user["id"]}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    readings = await db.sensor_readings.find({"machine_id": machine_id, "user_id": user["id"]}, {"_id": 0}).sort("timestamp", -1).to_list(1000)
    readings = readings[::-1]
    
    logs = await db.maintenance_logs.find({"machine_id": machine_id, "user_id": user["id"]}, {"_id": 0}).to_list(100)
    
    gnn_result = gnn_predict_pytorch(readings)
    nlp_score = nlp_predict(logs)
    health_score, _, _ = calculate_health_score(readings)
    
    prediction_data = multimodal_fusion_predict(gnn_result, nlp_score, health_score)
    
    prediction = Prediction(machine_id=machine_id, user_id=user["id"], timestamp=datetime.now(timezone.utc).isoformat(), **prediction_data)
    await db.predictions.insert_one(prediction.model_dump())
    
    new_failure_prob = round(prediction_data["fusion_score"] * 100, 1)
    new_risk_level = "critical" if prediction_data["fusion_score"] > 0.6 else "warning" if prediction_data["fusion_score"] > 0.3 else "healthy"
    
    await db.machines.update_one({"id": machine_id}, {"$set": {"health_score": health_score, "failure_probability": new_failure_prob, "risk_level": new_risk_level}})
    await check_and_create_alert(machine, user["id"], health_score, new_failure_prob, new_risk_level)
    
    return prediction

@api_router.get("/machines/{machine_id}/predictions", response_model=List[Prediction])
async def get_predictions(machine_id: str, user: dict = Depends(get_required_user)):
    predictions = await db.predictions.find({"machine_id": machine_id, "user_id": user["id"]}, {"_id": 0}).sort("timestamp", -1).to_list(50)
    return predictions

# Alert endpoints
@api_router.get("/alerts")
async def get_alerts(limit: int = 50, unacknowledged_only: bool = False, user: dict = Depends(get_required_user)):
    query = {"user_id": user["id"]}
    if unacknowledged_only:
        query["acknowledged"] = False
    alerts = await db.alerts.find(query, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return alerts

@api_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: dict = Depends(get_required_user)):
    result = await db.alerts.update_one({"id": alert_id, "user_id": user["id"]}, {"$set": {"acknowledged": True}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}

@api_router.get("/alert-settings")
async def get_alert_settings(user: dict = Depends(get_required_user)):
    settings = await db.alert_settings.find_one({"user_id": user["id"]}, {"_id": 0})
    if not settings:
        settings = AlertSettings(user_id=user["id"]).model_dump()
        await db.alert_settings.insert_one(settings)
    return settings

@api_router.put("/alert-settings")
async def update_alert_settings(update: AlertSettingsUpdate, user: dict = Depends(get_required_user)):
    update_dict = {k: v for k, v in update.model_dump().items() if v is not None}
    if update_dict:
        await db.alert_settings.update_one({"user_id": user["id"]}, {"$set": update_dict}, upsert=True)
    return await db.alert_settings.find_one({"user_id": user["id"]}, {"_id": 0})

# Dashboard summary
@api_router.get("/dashboard/summary")
async def get_dashboard_summary(user: dict = Depends(get_required_user)):
    machines = await db.machines.find({"user_id": user["id"]}, {"_id": 0}).to_list(100)
    
    total = len(machines)
    healthy = sum(1 for m in machines if m.get("risk_level") == "healthy")
    warning = sum(1 for m in machines if m.get("risk_level") == "warning")
    critical = sum(1 for m in machines if m.get("risk_level") == "critical")
    avg_health = np.mean([m.get("health_score", 100) for m in machines]) if machines else 100
    alert_count = await db.alerts.count_documents({"user_id": user["id"], "acknowledged": False})
    
    return {
        "total_machines": total,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "average_health_score": round(avg_health, 1),
        "unacknowledged_alerts": alert_count,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

# Seed demo data (creates user-specific demo data)
@api_router.post("/seed-demo")
async def seed_demo_data(user: dict = Depends(get_required_user)):
    user_id = user["id"]
    
    # Clear user's existing data
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
        machine = Machine(**m_data, user_id=user_id)
        await db.machines.insert_one(machine.model_dump())
        created_machines.append(machine)
        
        days = 30 + i * 15
        readings = simulate_sensor_data(machine.id, user_id, days)
        
        for j in range(0, len(readings), 500):
            await db.sensor_readings.insert_many(readings[j:j+500])
        
        health_score, failure_prob, risk_level = calculate_health_score(readings)
        await db.machines.update_one({"id": machine.id}, {"$set": {"health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}})
        
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
    
    return {"message": "Demo data seeded successfully", "machines_created": len(created_machines), "machine_ids": [m.id for m in created_machines]}

# File upload
@api_router.post("/upload")
async def upload_sensor_data(file: UploadFile = File(...), user: dict = Depends(get_required_user)):
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
                "user_id": user["id"],
                "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "temperature": float(row.get("temperature", row.get("temp", 0))),
                "pressure": float(row.get("pressure", 0)),
                "vibration": float(row.get("vibration", 0)),
                "rpm": float(row.get("rpm", 0)),
                "voltage": float(row.get("voltage", 0)) if row.get("voltage") else None,
                "current": float(row.get("current", 0)) if row.get("current") else None
            } for row in reader]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        for r in readings:
            r["user_id"] = user["id"]
        
        if readings:
            await db.sensor_readings.insert_many(readings)
        
        return {"message": f"Uploaded {len(readings)} readings", "count": len(readings)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{machine_id}")
async def websocket_endpoint(websocket: WebSocket, machine_id: str):
    await manager.connect(websocket, machine_id)
    
    try:
        machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
        if not machine:
            await websocket.close(code=4004)
            return
        
        user_id = machine.get("user_id", "")
        
        while True:
            reading = generate_live_reading(machine_id, user_id, machine.get("health_score", 50))
            await db.sensor_readings.insert_one(reading)
            
            recent = await db.sensor_readings.find({"machine_id": machine_id}, {"_id": 0}).sort("timestamp", -1).to_list(24)
            health_score, failure_prob, risk_level = calculate_health_score(recent[::-1])
            
            await db.machines.update_one({"id": machine_id}, {"$set": {"health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level}})
            
            machine_updated = await db.machines.find_one({"id": machine_id}, {"_id": 0})
            await check_and_create_alert(machine_updated, user_id, health_score, failure_prob, risk_level)
            
            await websocket.send_json({"type": "sensor_update", "reading": reading, "health_score": health_score, "failure_probability": failure_prob, "risk_level": risk_level})
            
            machine["health_score"] = health_score
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, machine_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, machine_id)

# Include router
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
