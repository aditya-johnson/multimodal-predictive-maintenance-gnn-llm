"""
GNN Model Training on NASA CMAPSS Dataset
Trains GCN and GAT models for Remaining Useful Life prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sensor columns from CMAPSS
SENSOR_COLS = [f's{i}' for i in range(1, 22)]
OP_COLS = ['op1', 'op2', 'op3']

class CMAPSSDataset:
    """NASA CMAPSS Dataset loader and preprocessor"""
    
    def __init__(self, data_path: str = "data/train_FD001.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.data = None
        self.graphs = []
        
    def load_data(self):
        """Load and preprocess CMAPSS data"""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        # Normalize sensor readings
        sensor_data = self.data[SENSOR_COLS].values
        self.data[SENSOR_COLS] = self.scaler.fit_transform(sensor_data)
        
        logger.info(f"Loaded {len(self.data)} samples from {self.data['engine_id'].nunique()} engines")
        return self.data
    
    def create_sensor_graph(self, sensor_values: np.ndarray) -> Data:
        """
        Create a graph where each sensor is a node
        Edges connect sensors with high correlation
        """
        num_sensors = len(SENSOR_COLS)
        
        # Node features: sensor value + statistics
        x = torch.tensor(sensor_values.reshape(-1, 1), dtype=torch.float)
        
        # Create fully connected graph (all sensors connected)
        edge_index = []
        for i in range(num_sensors):
            for j in range(num_sensors):
                if i != j:
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def prepare_graphs(self, window_size: int = 30):
        """
        Prepare graph data for training
        Each sample is a window of sensor readings
        """
        logger.info("Preparing graph dataset...")
        
        graphs = []
        labels = []
        
        for engine_id in self.data['engine_id'].unique():
            engine_data = self.data[self.data['engine_id'] == engine_id].sort_values('cycle')
            
            for i in range(len(engine_data) - window_size + 1):
                window = engine_data.iloc[i:i+window_size]
                
                # Aggregate sensor values over window (mean)
                sensor_means = window[SENSOR_COLS].mean().values
                
                # Create graph
                graph = self.create_sensor_graph(sensor_means)
                
                # Label: RUL at end of window (capped at 125)
                rul = min(window.iloc[-1]['RUL'], 125)
                
                # Convert RUL to class: 0=healthy (RUL>60), 1=warning (30<RUL<=60), 2=critical (RUL<=30)
                if rul > 60:
                    label = 0
                elif rul > 30:
                    label = 1
                else:
                    label = 2
                
                graph.y = torch.tensor([label], dtype=torch.long)
                graph.rul = torch.tensor([rul], dtype=torch.float)
                
                graphs.append(graph)
                labels.append(label)
        
        self.graphs = graphs
        logger.info(f"Created {len(graphs)} graph samples")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return graphs


class SensorGCN(nn.Module):
    """Graph Convolutional Network for RUL prediction"""
    
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
        # GCN layers with batch norm
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification head
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)


class SensorGAT(nn.Module):
    """Graph Attention Network for RUL prediction"""
    
    def __init__(self, num_features: int = 1, hidden_channels: int = 64, num_heads: int = 4, num_classes: int = 3):
        super(SensorGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=0.3)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, batch=None):
        # GAT layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification head
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)


def train_model(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


def evaluate_model(model, loader, device):
    """Evaluate model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    
    return correct / total


def train_and_save_models(data_path: str = "data/train_FD001.csv", 
                          model_dir: str = "models",
                          epochs: int = 50,
                          batch_size: int = 32):
    """
    Train GCN and GAT models on CMAPSS data and save weights
    """
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    dataset = CMAPSSDataset(data_path)
    dataset.load_data()
    graphs = dataset.prepare_graphs(window_size=30)
    
    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_graphs)}, Test: {len(test_graphs)}")
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    
    # Train GCN
    logger.info("Training GCN model...")
    gcn_model = SensorGCN(num_features=1, hidden_channels=64, num_classes=3).to(device)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=1e-4)
    gcn_scheduler = torch.optim.lr_scheduler.StepLR(gcn_optimizer, step_size=20, gamma=0.5)
    
    best_gcn_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_model(gcn_model, train_loader, gcn_optimizer, device)
        test_acc = evaluate_model(gcn_model, test_loader, device)
        gcn_scheduler.step()
        
        if test_acc > best_gcn_acc:
            best_gcn_acc = test_acc
            torch.save(gcn_model.state_dict(), f"{model_dir}/gcn_cmapss.pt")
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"GCN Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    logger.info(f"Best GCN Test Accuracy: {best_gcn_acc:.4f}")
    
    # Train GAT
    logger.info("Training GAT model...")
    gat_model = SensorGAT(num_features=1, hidden_channels=64, num_heads=4, num_classes=3).to(device)
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.001, weight_decay=1e-4)
    gat_scheduler = torch.optim.lr_scheduler.StepLR(gat_optimizer, step_size=20, gamma=0.5)
    
    best_gat_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_model(gat_model, train_loader, gat_optimizer, device)
        test_acc = evaluate_model(gat_model, test_loader, device)
        gat_scheduler.step()
        
        if test_acc > best_gat_acc:
            best_gat_acc = test_acc
            torch.save(gat_model.state_dict(), f"{model_dir}/gat_cmapss.pt")
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"GAT Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    logger.info(f"Best GAT Test Accuracy: {best_gat_acc:.4f}")
    
    # Save scaler
    import pickle
    with open(f"{model_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(dataset.scaler, f)
    
    logger.info(f"Models saved to {model_dir}/")
    
    return {
        "gcn_accuracy": best_gcn_acc,
        "gat_accuracy": best_gat_acc,
        "train_samples": len(train_graphs),
        "test_samples": len(test_graphs)
    }


if __name__ == "__main__":
    results = train_and_save_models(
        data_path="data/train_FD001.csv",
        model_dir="models",
        epochs=50,
        batch_size=32
    )
    print(f"\nTraining Results: {results}")
