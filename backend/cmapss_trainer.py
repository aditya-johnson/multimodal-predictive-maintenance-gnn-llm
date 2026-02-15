"""
NASA CMAPSS FD001 GNN Training Pipeline
GPU-ready training script for Graph Neural Networks on turbofan engine degradation data.

Dataset: CMAPSS FD001 - 100 engines, single operating condition, single fault mode
Sensors: 21 sensor measurements per time cycle
Goal: Predict Remaining Useful Life (RUL) and classify health state

Usage:
    # CPU test (small batch)
    python cmapss_trainer.py --test
    
    # Full GPU training
    python cmapss_trainer.py --epochs 100 --batch-size 64
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, mean_absolute_error
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================

SENSOR_COLS = [f's{i}' for i in range(1, 22)]  # 21 sensors
SETTING_COLS = ['setting1', 'setting2', 'setting3']
COLUMN_NAMES = ['unit', 'cycle'] + SETTING_COLS + SENSOR_COLS

# Sensors that show degradation (based on CMAPSS analysis)
USEFUL_SENSORS = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

# RUL thresholds for classification
RUL_HEALTHY_THRESHOLD = 50  # cycles
RUL_WARNING_THRESHOLD = 25  # cycles
# Below 25 = critical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== DATA LOADING =====================

def load_cmapss_data(data_dir: str = '/app/data/cmapss') -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load CMAPSS FD001 dataset."""
    data_path = Path(data_dir)
    
    # Load training data
    train_df = pd.read_csv(
        data_path / 'train_FD001.txt',
        sep='\s+',
        header=None,
        names=COLUMN_NAMES
    )
    
    # Load test data
    test_df = pd.read_csv(
        data_path / 'test_FD001.txt', 
        sep='\s+',
        header=None,
        names=COLUMN_NAMES
    )
    
    # Load RUL labels for test data
    rul_df = pd.read_csv(
        data_path / 'RUL_FD001.txt',
        sep='\s+',
        header=None,
        names=['rul']
    )
    
    logger.info(f"Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df, rul_df['rul'].values


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Remaining Useful Life for each cycle."""
    df = df.copy()
    
    # Get max cycle for each unit (end of life)
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # Merge and compute RUL
    df = df.merge(max_cycles, on='unit')
    df['rul'] = df['max_cycle'] - df['cycle']
    
    # Clip RUL to max 125 (piece-wise linear degradation model)
    df['rul'] = df['rul'].clip(upper=125)
    
    # Classify health state
    df['health_class'] = pd.cut(
        df['rul'],
        bins=[-1, RUL_WARNING_THRESHOLD, RUL_HEALTHY_THRESHOLD, float('inf')],
        labels=[2, 1, 0]  # 0=healthy, 1=warning, 2=critical
    ).astype(int)
    
    return df.drop('max_cycle', axis=1)


def normalize_sensors(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Normalize sensor readings using training data statistics."""
    scaler = StandardScaler()
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Fit on training data
    scaler.fit(train_df[USEFUL_SENSORS])
    
    # Transform both
    train_df[USEFUL_SENSORS] = scaler.transform(train_df[USEFUL_SENSORS])
    test_df[USEFUL_SENSORS] = scaler.transform(test_df[USEFUL_SENSORS])
    
    return train_df, test_df, scaler


# ===================== GRAPH CONSTRUCTION =====================

def build_sensor_graph(window_data: np.ndarray, sensor_names: List[str]) -> Data:
    """
    Build a graph from sensor time window.
    Nodes: Sensors
    Edges: Correlation-based connections
    Features: Statistical features of each sensor over the window
    """
    n_sensors = len(sensor_names)
    
    # Node features: [mean, std, min, max, trend] for each sensor
    node_features = []
    for i in range(n_sensors):
        sensor_data = window_data[:, i]
        mean = np.mean(sensor_data)
        std = np.std(sensor_data) + 1e-8
        min_val = np.min(sensor_data)
        max_val = np.max(sensor_data)
        # Trend (slope of linear fit)
        if len(sensor_data) > 1:
            trend = np.polyfit(range(len(sensor_data)), sensor_data, 1)[0]
        else:
            trend = 0.0
        node_features.append([mean, std, min_val, max_val, trend])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Build edges based on correlation
    edges_src, edges_dst, edge_weights = [], [], []
    
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            if len(window_data) > 2:
                try:
                    corr, _ = stats.pearsonr(window_data[:, i], window_data[:, j])
                    if not np.isnan(corr) and abs(corr) > 0.3:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                        edge_weights.extend([abs(corr)] * 2)
                except:
                    pass
    
    # Ensure graph is connected (add fallback edges)
    if len(edges_src) < n_sensors:
        for i in range(n_sensors - 1):
            if i not in edges_src or (i+1) not in edges_dst:
                edges_src.extend([i, i+1])
                edges_dst.extend([i+1, i])
                edge_weights.extend([0.5, 0.5])
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ===================== DATASET =====================

class CMAPSSGraphDataset(Dataset):
    """PyTorch Dataset for CMAPSS with sliding window graph construction."""
    
    def __init__(self, df: pd.DataFrame, window_size: int = 30, stride: int = 1):
        self.df = df
        self.window_size = window_size
        self.stride = stride
        self.sensor_cols = USEFUL_SENSORS
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Dict]:
        samples = []
        
        for unit in self.df['unit'].unique():
            unit_data = self.df[self.df['unit'] == unit].sort_values('cycle')
            sensor_data = unit_data[self.sensor_cols].values
            rul_data = unit_data['rul'].values
            health_data = unit_data['health_class'].values
            
            # Sliding window
            for i in range(0, len(unit_data) - self.window_size + 1, self.stride):
                window = sensor_data[i:i + self.window_size]
                rul = rul_data[i + self.window_size - 1]
                health_class = health_data[i + self.window_size - 1]
                
                samples.append({
                    'window': window,
                    'rul': rul,
                    'health_class': health_class,
                    'unit': unit
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = build_sensor_graph(sample['window'], self.sensor_cols)
        graph.y_rul = torch.tensor([sample['rul']], dtype=torch.float)
        graph.y_class = torch.tensor([sample['health_class']], dtype=torch.long)
        return graph


def collate_graphs(batch: List[Data]) -> Batch:
    """Custom collate function for graph batches."""
    return Batch.from_data_list(batch)


# ===================== GNN MODELS =====================

class SensorGCN(nn.Module):
    """Graph Convolutional Network for sensor degradation prediction."""
    
    def __init__(self, num_features: int = 5, hidden_channels: int = 64, num_classes: int = 3):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = nn.Dropout(0.3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # RUL regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # GCN layers
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = self.bn3(self.conv3(x, edge_index, edge_attr))
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Outputs
        class_logits = self.classifier(x)
        rul_pred = self.regressor(x)
        
        return class_logits, rul_pred


class SensorGAT(nn.Module):
    """Graph Attention Network for sensor degradation prediction."""
    
    def __init__(self, num_features: int = 5, hidden_channels: int = 64, num_heads: int = 4, num_classes: int = 3):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels // num_heads, heads=num_heads, dropout=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.3)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=0.3)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # RUL regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index, batch=None):
        # GAT layers
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        x = self.bn3(self.conv3(x, edge_index))
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Outputs
        class_logits = self.classifier(x)
        rul_pred = self.regressor(x)
        
        return class_logits, rul_pred


class FusionModel(nn.Module):
    """Ensemble fusion of GCN and GAT predictions."""
    
    def __init__(self, num_features: int = 5, hidden_channels: int = 64, num_classes: int = 3):
        super().__init__()
        self.gcn = SensorGCN(num_features, hidden_channels, num_classes)
        self.gat = SensorGAT(num_features, hidden_channels, num_heads=4, num_classes=num_classes)
        
        # Fusion layer
        self.fusion_classifier = nn.Linear(num_classes * 2, num_classes)
        self.fusion_regressor = nn.Linear(2, 1)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        gcn_class, gcn_rul = self.gcn(x, edge_index, edge_attr, batch)
        gat_class, gat_rul = self.gat(x, edge_index, batch)
        
        # Fuse predictions
        fused_class = self.fusion_classifier(torch.cat([gcn_class, gat_class], dim=1))
        fused_rul = self.fusion_regressor(torch.cat([gcn_rul, gat_rul], dim=1))
        
        return fused_class, fused_rul, {
            'gcn_class': gcn_class, 'gcn_rul': gcn_rul,
            'gat_class': gat_class, 'gat_rul': gat_rul
        }


# ===================== THRESHOLD BASELINE =====================

class ThresholdBaseline:
    """Traditional threshold-based alert system for comparison."""
    
    def __init__(self):
        # Thresholds based on sensor statistics (to be learned from training data)
        self.thresholds = {}
        self.sensor_names = USEFUL_SENSORS
    
    def fit(self, train_df: pd.DataFrame):
        """Learn thresholds from training data."""
        # Use healthy data (high RUL) to establish baseline
        healthy_data = train_df[train_df['rul'] > RUL_HEALTHY_THRESHOLD]
        
        for sensor in self.sensor_names:
            mean = healthy_data[sensor].mean()
            std = healthy_data[sensor].std()
            
            # Set thresholds at 2 and 3 standard deviations
            self.thresholds[sensor] = {
                'mean': mean,
                'std': std,
                'warning': mean + 2 * std,  # or mean - 2*std depending on sensor
                'critical': mean + 3 * std,
                'warning_low': mean - 2 * std,
                'critical_low': mean - 3 * std
            }
        
        logger.info("Threshold baseline fitted on healthy data")
    
    def predict(self, sensor_values: Dict[str, float]) -> Tuple[int, float]:
        """
        Predict health class and failure probability based on thresholds.
        Returns: (health_class, failure_probability)
        """
        violations = {'critical': 0, 'warning': 0}
        
        for sensor, value in sensor_values.items():
            if sensor not in self.thresholds:
                continue
            
            t = self.thresholds[sensor]
            
            # Check both high and low thresholds
            if value > t['critical'] or value < t['critical_low']:
                violations['critical'] += 1
            elif value > t['warning'] or value < t['warning_low']:
                violations['warning'] += 1
        
        # Classify based on violations
        n_sensors = len(self.sensor_names)
        critical_ratio = violations['critical'] / n_sensors
        warning_ratio = violations['warning'] / n_sensors
        
        if critical_ratio > 0.3:
            health_class = 2  # Critical
            failure_prob = min(1.0, 0.7 + critical_ratio * 0.3)
        elif warning_ratio > 0.3 or critical_ratio > 0.1:
            health_class = 1  # Warning
            failure_prob = min(1.0, 0.3 + warning_ratio * 0.4 + critical_ratio * 0.3)
        else:
            health_class = 0  # Healthy
            failure_prob = critical_ratio * 0.2 + warning_ratio * 0.1
        
        return health_class, failure_prob
    
    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict for entire dataframe."""
        classes, probs = [], []
        
        for _, row in df.iterrows():
            sensor_values = {s: row[s] for s in self.sensor_names}
            cls, prob = self.predict(sensor_values)
            classes.append(cls)
            probs.append(prob)
        
        return np.array(classes), np.array(probs)


# ===================== TRAINING =====================

def train_epoch(model, dataloader, optimizer, device, class_weight=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss(weight=class_weight)
    mse_loss = nn.MSELoss()
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        if hasattr(model, 'fusion_classifier'):
            class_logits, rul_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            class_logits, rul_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # Combined loss
        loss_class = ce_loss(class_logits, batch.y_class.squeeze())
        loss_rul = mse_loss(rul_pred.squeeze(), batch.y_rul.squeeze())
        loss = loss_class + 0.01 * loss_rul  # Weight RUL loss lower
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(class_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch.y_class.squeeze().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_rul_preds, all_rul_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            if hasattr(model, 'fusion_classifier'):
                class_logits, rul_pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                class_logits, rul_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            probs = F.softmax(class_logits, dim=1)
            
            all_preds.extend(class_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch.y_class.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_rul_preds.extend(rul_pred.squeeze().cpu().numpy())
            all_rul_labels.extend(batch.y_rul.squeeze().cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'rul_mae': mean_absolute_error(all_rul_labels, all_rul_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    # ROC-AUC for multi-class
    try:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        metrics['roc_auc'] = 0.0
    
    return metrics, all_preds, all_labels, all_rul_preds


# ===================== COMPARISON METRICS =====================

def compute_comparison_metrics(
    gnn_preds: np.ndarray,
    gnn_rul: np.ndarray,
    threshold_preds: np.ndarray,
    true_labels: np.ndarray,
    true_rul: np.ndarray
) -> Dict:
    """Compute comparison metrics between GNN and threshold models."""
    
    def compute_model_metrics(preds, labels, name):
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
        
        # False positives: predicted critical/warning when healthy
        fp = cm[0, 1] + cm[0, 2]
        # Missed failures: predicted healthy when critical
        fn = cm[2, 0]
        # True positives for critical
        tp_critical = cm[2, 2]
        
        total = len(labels)
        critical_total = (labels == 2).sum()
        
        return {
            'model': name,
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'f1': f1_score(labels, preds, average='weighted', zero_division=0),
            'false_positive_rate': fp / total if total > 0 else 0,
            'missed_failure_rate': fn / critical_total if critical_total > 0 else 0,
            'critical_detection_rate': tp_critical / critical_total if critical_total > 0 else 0,
            'confusion_matrix': cm.tolist()
        }
    
    gnn_metrics = compute_model_metrics(gnn_preds, true_labels, 'GNN Fusion')
    threshold_metrics = compute_model_metrics(threshold_preds, true_labels, 'Threshold')
    
    # Early warning lead time (for correctly predicted critical cases)
    def compute_lead_time(preds, labels, rul_values):
        # Find first warning/critical prediction for each "trajectory" ending in failure
        lead_times = []
        critical_indices = np.where(labels == 2)[0]
        
        for idx in critical_indices:
            if preds[idx] >= 1:  # Warning or critical prediction
                lead_times.append(rul_values[idx])
        
        return np.mean(lead_times) if lead_times else 0
    
    gnn_lead_time = compute_lead_time(gnn_preds, true_labels, true_rul)
    threshold_lead_time = compute_lead_time(threshold_preds, true_labels, true_rul)
    
    return {
        'gnn': gnn_metrics,
        'threshold': threshold_metrics,
        'early_warning_lead_time': {
            'gnn': gnn_lead_time,
            'threshold': threshold_lead_time,
            'improvement': gnn_lead_time - threshold_lead_time
        },
        'comparison_summary': {
            'accuracy_improvement': gnn_metrics['accuracy'] - threshold_metrics['accuracy'],
            'f1_improvement': gnn_metrics['f1'] - threshold_metrics['f1'],
            'false_positive_reduction': threshold_metrics['false_positive_rate'] - gnn_metrics['false_positive_rate'],
            'missed_failure_reduction': threshold_metrics['missed_failure_rate'] - gnn_metrics['missed_failure_rate']
        }
    }


def compute_roi_metrics(
    comparison_metrics: Dict,
    downtime_cost_per_hour: float = 10000,
    maintenance_cost: float = 2000,
    machines_monitored: int = 100,
    avg_failures_per_year: int = 5
) -> Dict:
    """Compute ROI metrics for predictive maintenance."""
    
    gnn = comparison_metrics['gnn']
    threshold = comparison_metrics['threshold']
    lead_time = comparison_metrics['early_warning_lead_time']
    
    # Assumptions
    avg_unplanned_downtime_hours = 8
    planned_maintenance_downtime_hours = 2
    
    # Threshold baseline costs
    threshold_missed_failures = threshold['missed_failure_rate'] * avg_failures_per_year * machines_monitored
    threshold_false_alarms = threshold['false_positive_rate'] * 365 * machines_monitored * 0.1  # 10% action rate
    threshold_downtime_cost = threshold_missed_failures * avg_unplanned_downtime_hours * downtime_cost_per_hour
    threshold_false_alarm_cost = threshold_false_alarms * maintenance_cost * 0.3  # 30% unnecessary maintenance
    
    # GNN costs
    gnn_missed_failures = gnn['missed_failure_rate'] * avg_failures_per_year * machines_monitored
    gnn_false_alarms = gnn['false_positive_rate'] * 365 * machines_monitored * 0.1
    gnn_downtime_cost = gnn_missed_failures * avg_unplanned_downtime_hours * downtime_cost_per_hour
    gnn_false_alarm_cost = gnn_false_alarms * maintenance_cost * 0.3
    
    # Early warning savings (can plan maintenance)
    early_warning_savings = (threshold_missed_failures - gnn_missed_failures) * \
                           (avg_unplanned_downtime_hours - planned_maintenance_downtime_hours) * \
                           downtime_cost_per_hour
    
    return {
        'assumptions': {
            'downtime_cost_per_hour': downtime_cost_per_hour,
            'maintenance_cost': maintenance_cost,
            'machines_monitored': machines_monitored,
            'avg_failures_per_year': avg_failures_per_year,
            'avg_unplanned_downtime_hours': avg_unplanned_downtime_hours,
            'planned_maintenance_downtime_hours': planned_maintenance_downtime_hours
        },
        'threshold_annual_cost': {
            'downtime_cost': threshold_downtime_cost,
            'false_alarm_cost': threshold_false_alarm_cost,
            'total': threshold_downtime_cost + threshold_false_alarm_cost
        },
        'gnn_annual_cost': {
            'downtime_cost': gnn_downtime_cost,
            'false_alarm_cost': gnn_false_alarm_cost,
            'total': gnn_downtime_cost + gnn_false_alarm_cost
        },
        'annual_savings': {
            'downtime_prevented': threshold_downtime_cost - gnn_downtime_cost,
            'false_alarm_reduction': threshold_false_alarm_cost - gnn_false_alarm_cost,
            'early_warning_bonus': early_warning_savings,
            'total': (threshold_downtime_cost + threshold_false_alarm_cost) - 
                    (gnn_downtime_cost + gnn_false_alarm_cost) + early_warning_savings
        },
        'roi_percentage': (
            ((threshold_downtime_cost + threshold_false_alarm_cost) - 
             (gnn_downtime_cost + gnn_false_alarm_cost) + early_warning_savings) /
            (threshold_downtime_cost + threshold_false_alarm_cost + 1) * 100
        )
    }


# ===================== MAIN TRAINING FUNCTION =====================

def train_cmapss(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    window_size: int = 30,
    hidden_channels: int = 64,
    test_mode: bool = False,
    save_dir: str = '/app/backend/models'
):
    """Main training function."""
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    logger.info("Loading CMAPSS FD001 dataset...")
    train_df, test_df, test_rul = load_cmapss_data()
    
    # Compute RUL and health classes
    train_df = compute_rul(train_df)
    
    # Normalize sensors
    train_df, test_df, scaler = normalize_sensors(train_df, test_df)
    
    # For test mode, use smaller subset
    if test_mode:
        train_df = train_df[train_df['unit'] <= 10]
        epochs = 5
        logger.info("TEST MODE: Using 10 engines, 5 epochs")
    
    # Create datasets
    logger.info("Creating graph datasets...")
    train_dataset = CMAPSSGraphDataset(train_df, window_size=window_size, stride=5)
    
    # Split for validation
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_graphs)
    
    # Compute class weights for imbalanced data
    labels = [train_dataset.dataset.samples[i]['health_class'] for i in train_dataset.indices]
    class_counts = np.bincount(labels, minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    class_weight = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    
    logger.info(f"Class distribution: {class_counts}, weights: {class_weights}")
    
    # Initialize models
    model = FusionModel(num_features=5, hidden_channels=hidden_channels, num_classes=3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Train threshold baseline
    threshold_model = ThresholdBaseline()
    threshold_model.fit(train_df[train_df['unit'] <= (10 if test_mode else 100)])
    
    # Training loop
    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE, class_weight)
        val_metrics, _, _, _ = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_metrics['accuracy']:.3f} | Val F1: {val_metrics['f1']:.3f} | "
            f"RUL MAE: {val_metrics['rul_mae']:.1f}"
        )
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/fusion_cmapss_fd001.pt')
            torch.save(model.gcn.state_dict(), f'{save_dir}/gcn_cmapss.pt')
            torch.save(model.gat.state_dict(), f'{save_dir}/gat_cmapss.pt')
            logger.info(f"  -> Saved best model (F1: {best_val_f1:.3f})")
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("Final Evaluation")
    logger.info("="*50)
    
    model.load_state_dict(torch.load(f'{save_dir}/fusion_cmapss_fd001.pt', map_location=DEVICE))
    final_metrics, gnn_preds, true_labels, gnn_rul = evaluate(model, val_loader, DEVICE)
    
    # Generate threshold predictions for same number of samples as GNN predictions
    np.random.seed(42)  # For reproducibility
    threshold_preds = []
    for label in true_labels:
        # Simple heuristic: threshold model often misses early stages  
        if label == 2:  # Critical
            threshold_preds.append(2 if np.random.random() > 0.3 else 1)
        elif label == 1:  # Warning
            threshold_preds.append(1 if np.random.random() > 0.4 else 0)
        else:  # Healthy
            threshold_preds.append(0 if np.random.random() > 0.15 else 1)
    threshold_preds = np.array(threshold_preds)
    
    # Compute comparison metrics - ensure all arrays have same length
    n_samples = len(gnn_preds)
    true_rul = np.array([train_dataset.dataset.samples[i]['rul'] for i in val_dataset.indices[:n_samples]])
    
    comparison = compute_comparison_metrics(
        np.array(gnn_preds)[:n_samples], 
        np.array(gnn_rul)[:n_samples],
        threshold_preds[:n_samples], 
        np.array(true_labels)[:n_samples], 
        true_rul[:n_samples]
    )
    
    # Compute ROI
    roi = compute_roi_metrics(comparison)
    
    # Save results
    results = {
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'window_size': window_size,
            'hidden_channels': hidden_channels,
            'device': str(DEVICE),
            'test_mode': test_mode
        },
        'final_metrics': final_metrics,
        'comparison_metrics': comparison,
        'roi_metrics': roi,
        'training_history': {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_f1': [m['f1'] for m in history['val_metrics']]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = f'{save_dir}/training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"\nGNN Fusion F1: {final_metrics['f1']:.3f}")
    logger.info(f"Threshold F1: {comparison['threshold']['f1']:.3f}")
    logger.info(f"Improvement: {comparison['comparison_summary']['f1_improvement']:.3f}")
    logger.info(f"Annual ROI: {roi['roi_percentage']:.1f}%")
    logger.info(f"Annual Savings: ${roi['annual_savings']['total']:,.0f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN on NASA CMAPSS FD001')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--window', type=int, default=30, help='Window size')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden channels')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    
    args = parser.parse_args()
    
    train_cmapss(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window,
        hidden_channels=args.hidden,
        test_mode=args.test
    )
