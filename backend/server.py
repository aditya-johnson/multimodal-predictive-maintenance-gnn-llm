from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import numpy as np
import json
import io
import csv
from scipy import stats
import networkx as nx

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Multimodal Predictive Maintenance API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class SensorGraph(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

# ===================== DATA SIMULATION =====================

def generate_degradation_pattern(days: int, failure_point: float = 0.8) -> np.ndarray:
    """Generate realistic degradation pattern."""
    x = np.linspace(0, 1, days)
    # Exponential degradation with noise
    degradation = 1 - np.exp(-3 * (x ** 2))
    noise = np.random.normal(0, 0.02, days)
    return np.clip(degradation + noise, 0, 1)

def simulate_sensor_data(machine_id: str, days: int = 90) -> List[dict]:
    """Simulate run-to-failure sensor data."""
    readings = []
    degradation = generate_degradation_pattern(days)
    
    # Base sensor values
    base_temp = 45 + np.random.uniform(-5, 5)
    base_pressure = 100 + np.random.uniform(-10, 10)
    base_vibration = 0.5 + np.random.uniform(-0.1, 0.1)
    base_rpm = 3000 + np.random.uniform(-100, 100)
    
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    
    for i in range(days * 24):  # Hourly readings
        d = degradation[min(i // 24, days - 1)]
        
        # Sensor values degrade over time with correlations
        temp = base_temp + d * 35 + np.random.normal(0, 2)
        pressure = base_pressure - d * 25 + np.random.normal(0, 3)
        vibration = base_vibration + d * 4.5 + np.random.normal(0, 0.1)
        rpm = base_rpm - d * 500 + np.random.normal(0, 50)
        
        # Add anomalies
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

def calculate_health_score(readings: List[dict]) -> tuple:
    """Calculate health score from recent sensor readings."""
    if not readings:
        return 100.0, 0.0, "healthy"
    
    recent = readings[-24:]  # Last 24 hours
    
    # Weighted score based on sensor thresholds
    temps = [r["temperature"] for r in recent]
    vibs = [r["vibration"] for r in recent]
    pressures = [r["pressure"] for r in recent]
    
    temp_score = max(0, 100 - (np.mean(temps) - 45) * 2)
    vib_score = max(0, 100 - np.mean(vibs) * 20)
    pressure_score = max(0, min(100, np.mean(pressures)))
    
    health_score = (temp_score * 0.3 + vib_score * 0.4 + pressure_score * 0.3)
    health_score = max(0, min(100, health_score))
    
    # Failure probability
    failure_prob = max(0, min(100, (100 - health_score) * 1.2))
    
    # Risk level
    if health_score >= 70:
        risk_level = "healthy"
    elif health_score >= 40:
        risk_level = "warning"
    else:
        risk_level = "critical"
    
    return round(health_score, 1), round(failure_prob, 1), risk_level

# ===================== GRAPH NEURAL NETWORK =====================

def build_sensor_correlation_graph(readings: List[dict]) -> Dict:
    """Build correlation graph between sensors."""
    if len(readings) < 10:
        # Return default graph structure
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
    
    for r in readings[-500:]:  # Use recent readings
        for s in sensors:
            if s in r:
                data[s].append(r[s])
    
    # Calculate correlations
    nodes = []
    links = []
    
    for i, s in enumerate(sensors):
        # Node size based on variance (anomaly indicator)
        variance = np.var(data[s]) if data[s] else 0
        nodes.append({
            "id": s,
            "group": (i % 3) + 1,
            "value": min(100, variance * 10)
        })
    
    # Calculate edge weights from correlations
    for i, s1 in enumerate(sensors):
        for j, s2 in enumerate(sensors):
            if i < j and len(data[s1]) == len(data[s2]) and len(data[s1]) > 2:
                corr, _ = stats.pearsonr(data[s1], data[s2])
                if abs(corr) > 0.3:  # Threshold for significant correlation
                    links.append({
                        "source": s1,
                        "target": s2,
                        "weight": round(abs(corr), 3)
                    })
    
    return {"nodes": nodes, "links": links}

def gnn_predict(graph: Dict, readings: List[dict]) -> float:
    """Simplified GNN-like prediction using graph features."""
    if not readings:
        return 0.5
    
    # Feature extraction from graph
    total_edge_weight = sum(l.get("weight", 0) for l in graph.get("links", []))
    node_variance = sum(n.get("value", 0) for n in graph.get("nodes", []))
    
    # Recent sensor trends
    recent = readings[-100:]
    if not recent:
        return 0.5
    
    temp_trend = (recent[-1]["temperature"] - recent[0]["temperature"]) / max(1, recent[0]["temperature"])
    vib_trend = (recent[-1]["vibration"] - recent[0]["vibration"]) / max(0.1, recent[0]["vibration"])
    
    # Combine features for prediction
    score = 0.3 * min(1, total_edge_weight / 3) + \
            0.2 * min(1, node_variance / 200) + \
            0.25 * min(1, max(0, temp_trend)) + \
            0.25 * min(1, max(0, vib_trend))
    
    return round(score, 3)

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
    
    # Calculate embedding similarity score (simplified)
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

def multimodal_fusion_predict(gnn_score: float, nlp_score: float, health_score: float) -> dict:
    """Fuse GNN and NLP predictions for final output."""
    # Weighted fusion
    fusion_score = 0.4 * gnn_score + 0.3 * nlp_score + 0.3 * (1 - health_score / 100)
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
        "gnn_score": gnn_score,
        "nlp_score": nlp_score,
        "fusion_score": round(fusion_score, 3)
    }

# ===================== API ENDPOINTS =====================

@api_router.get("/")
async def root():
    return {"message": "Multimodal Predictive Maintenance API", "status": "operational"}

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
    # Also delete related data
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
    return readings[::-1]  # Return in chronological order

@api_router.post("/machines/{machine_id}/simulate")
async def simulate_machine_data(machine_id: str, days: int = 90):
    """Simulate sensor data for a machine."""
    machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    # Generate simulated data
    readings = simulate_sensor_data(machine_id, days)
    
    # Insert readings in batches
    batch_size = 500
    for i in range(0, len(readings), batch_size):
        batch = readings[i:i+batch_size]
        await db.sensor_readings.insert_many(batch)
    
    # Update machine health score
    health_score, failure_prob, risk_level = calculate_health_score(readings)
    await db.machines.update_one(
        {"id": machine_id},
        {"$set": {
            "health_score": health_score,
            "failure_probability": failure_prob,
            "risk_level": risk_level
        }}
    )
    
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
    """Run multimodal prediction for a machine."""
    machine = await db.machines.find_one({"id": machine_id}, {"_id": 0})
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    
    # Get sensor readings
    readings = await db.sensor_readings.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(1000)
    readings = readings[::-1]
    
    # Get maintenance logs
    logs = await db.maintenance_logs.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).to_list(100)
    
    # Build graph and get GNN prediction
    graph = build_sensor_correlation_graph(readings)
    gnn_score = gnn_predict(graph, readings)
    
    # Get NLP prediction
    nlp_score = nlp_predict(logs)
    
    # Get health score
    health_score, _, _ = calculate_health_score(readings)
    
    # Multimodal fusion
    prediction_data = multimodal_fusion_predict(gnn_score, nlp_score, health_score)
    
    prediction = Prediction(
        machine_id=machine_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **prediction_data
    )
    
    doc = prediction.model_dump()
    await db.predictions.insert_one(doc)
    
    # Update machine with latest prediction
    await db.machines.update_one(
        {"id": machine_id},
        {"$set": {
            "health_score": health_score,
            "failure_probability": round(prediction_data["fusion_score"] * 100, 1),
            "risk_level": "critical" if prediction_data["fusion_score"] > 0.6 else 
                         "warning" if prediction_data["fusion_score"] > 0.3 else "healthy"
        }}
    )
    
    return prediction

@api_router.get("/machines/{machine_id}/predictions", response_model=List[Prediction])
async def get_predictions(machine_id: str):
    predictions = await db.predictions.find(
        {"machine_id": machine_id}, {"_id": 0}
    ).sort("timestamp", -1).to_list(50)
    return predictions

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
    
    return {
        "total_machines": total_machines,
        "healthy": healthy,
        "warning": warning,
        "critical": critical,
        "average_health_score": round(avg_health, 1),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

# Seed demo data endpoint
@api_router.post("/seed-demo")
async def seed_demo_data():
    """Seed database with demo machines and data."""
    # Clear existing data
    await db.machines.delete_many({})
    await db.sensor_readings.delete_many({})
    await db.maintenance_logs.delete_many({})
    await db.predictions.delete_many({})
    
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
        
        # Simulate sensor data with varying degradation
        days = 30 + i * 15  # Different degradation stages
        readings = simulate_sensor_data(machine.id, days)
        
        # Insert in batches
        batch_size = 500
        for j in range(0, len(readings), batch_size):
            batch = readings[j:j+batch_size]
            await db.sensor_readings.insert_many(batch)
        
        # Calculate and update health score
        health_score, failure_prob, risk_level = calculate_health_score(readings)
        await db.machines.update_one(
            {"id": machine.id},
            {"$set": {
                "health_score": health_score,
                "failure_probability": failure_prob,
                "risk_level": risk_level
            }}
        )
        
        # Add maintenance logs
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
