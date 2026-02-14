import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, LogOut, User } from "lucide-react";

import { AuthPage } from "./components/AuthPage";
import { Sidebar } from "./components/Sidebar";
import { HealthDashboard } from "./components/HealthDashboard";
import { SensorTimeSeries } from "./components/SensorTimeSeries";
import { FailurePrediction } from "./components/FailurePrediction";
import { GraphVisualization } from "./components/GraphVisualization";
import { MaintenanceLogs } from "./components/MaintenanceLogs";
import { AlertsPanel } from "./components/AlertsPanel";
import { useWebSocket } from "./hooks/useWebSocket";
import { Button } from "./components/ui/button";

import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Configure axios defaults
axios.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [machines, setMachines] = useState([]);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [loading, setLoading] = useState(true);
  const [dashboardSummary, setDashboardSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);

  // Check for existing session
  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    const storedUser = localStorage.getItem("user");
    
    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  // WebSocket for real-time updates
  const { lastMessage, isConnected } = useWebSocket(
    selectedMachine?.id && token
      ? `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/${selectedMachine.id}`
      : null
  );

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === "sensor_update") {
        setMachines(prev => prev.map(m => 
          m.id === selectedMachine?.id 
            ? { ...m, health_score: lastMessage.health_score, failure_probability: lastMessage.failure_probability, risk_level: lastMessage.risk_level }
            : m
        ));
        
        if (selectedMachine) {
          setSelectedMachine(prev => ({
            ...prev,
            health_score: lastMessage.health_score,
            failure_probability: lastMessage.failure_probability,
            risk_level: lastMessage.risk_level
          }));
        }
      } else if (lastMessage.type === "alert") {
        const alert = lastMessage.data;
        toast.error(`${alert.alert_type}: ${alert.machine_name}`, {
          description: alert.message,
          duration: 10000
        });
        setAlerts(prev => [alert, ...prev]);
        fetchAlerts();
      }
    }
  }, [lastMessage, selectedMachine]);

  const fetchMachines = useCallback(async () => {
    if (!token) return;
    try {
      const response = await axios.get(`${API}/machines`);
      setMachines(response.data);
      if (response.data.length > 0 && !selectedMachine) {
        setSelectedMachine(response.data[0]);
      }
    } catch (error) {
      console.error("Error fetching machines:", error);
    }
  }, [token, selectedMachine]);

  const fetchDashboardSummary = async () => {
    if (!token) return;
    try {
      const response = await axios.get(`${API}/dashboard/summary`);
      setDashboardSummary(response.data);
    } catch (error) {
      console.error("Error fetching summary:", error);
    }
  };

  const fetchAlerts = async () => {
    if (!token) return;
    try {
      const response = await axios.get(`${API}/alerts?limit=20&unacknowledged_only=true`);
      setAlerts(response.data);
    } catch (error) {
      console.error("Error fetching alerts:", error);
    }
  };

  const acknowledgeAlert = async (alertId) => {
    try {
      await axios.post(`${API}/alerts/${alertId}/acknowledge`);
      setAlerts(prev => prev.filter(a => a.id !== alertId));
      toast.success("Alert acknowledged");
      fetchDashboardSummary();
    } catch (error) {
      toast.error("Failed to acknowledge alert");
    }
  };

  const seedDemoData = async () => {
    setLoading(true);
    try {
      toast.loading("Seeding demo data...", { id: "seed" });
      await axios.post(`${API}/seed-demo`);
      toast.success("Demo data loaded successfully!", { id: "seed" });
      await fetchMachines();
      await fetchDashboardSummary();
      await fetchAlerts();
    } catch (error) {
      toast.error("Failed to seed demo data", { id: "seed" });
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async (machineId) => {
    try {
      toast.loading("Running GNN + NLP prediction...", { id: "predict" });
      await axios.post(`${API}/machines/${machineId}/predict`);
      toast.success("Prediction complete!", { id: "predict" });
      await fetchMachines();
      await fetchDashboardSummary();
      await fetchAlerts();
    } catch (error) {
      toast.error("Prediction failed", { id: "predict" });
    }
  };

  const handleLogin = (userData, accessToken) => {
    setUser(userData);
    setToken(accessToken);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
    setToken(null);
    setMachines([]);
    setSelectedMachine(null);
    toast.success("Logged out successfully");
  };

  // Fetch data when authenticated
  useEffect(() => {
    if (token) {
      const init = async () => {
        setLoading(true);
        await fetchMachines();
        await fetchDashboardSummary();
        await fetchAlerts();
        setLoading(false);
      };
      init();
    }
  }, [token]);

  // Show auth page if not logged in
  if (!token) {
    return (
      <>
        <Toaster theme="dark" position="top-right" />
        <AuthPage onLogin={handleLogin} API={API} />
      </>
    );
  }

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":
        return (
          <HealthDashboard
            machines={machines}
            selectedMachine={selectedMachine}
            setSelectedMachine={setSelectedMachine}
            dashboardSummary={dashboardSummary}
            onRunPrediction={runPrediction}
            API={API}
            isStreaming={isConnected}
          />
        );
      case "sensors":
        return (
          <SensorTimeSeries
            selectedMachine={selectedMachine}
            API={API}
            isStreaming={isConnected}
            lastReading={lastMessage?.type === "sensor_update" ? lastMessage.reading : null}
          />
        );
      case "prediction":
        return (
          <FailurePrediction
            selectedMachine={selectedMachine}
            onRunPrediction={runPrediction}
            API={API}
          />
        );
      case "graph":
        return (
          <GraphVisualization
            selectedMachine={selectedMachine}
            API={API}
          />
        );
      case "logs":
        return (
          <MaintenanceLogs
            selectedMachine={selectedMachine}
            machines={machines}
            API={API}
          />
        );
      case "alerts":
        return (
          <AlertsPanel
            alerts={alerts}
            onAcknowledge={acknowledgeAlert}
            API={API}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-zinc-950 overflow-hidden" data-testid="app-container">
      <Toaster 
        theme="dark" 
        position="top-right"
        toastOptions={{
          style: {
            background: '#18181b',
            border: '1px solid #27272a',
            color: '#e4e4e7'
          }
        }}
      />
      
      <Sidebar 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        machines={machines}
        selectedMachine={selectedMachine}
        setSelectedMachine={setSelectedMachine}
        onSeedDemo={seedDemoData}
        loading={loading}
        alertCount={alerts.length}
      />
      
      <main className="flex-1 overflow-y-auto flex flex-col">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-zinc-800/60 bg-zinc-900/30">
          <div className="flex items-center gap-2">
            {isConnected && (
              <span className="flex items-center gap-2 text-xs text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Live Streaming
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <User className="w-4 h-4" />
              <span>{user?.name || user?.email}</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleLogout}
              className="text-zinc-400 hover:text-zinc-100"
              data-testid="logout-btn"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6 md:p-8 overflow-y-auto">
          {/* Alert Banner */}
          {alerts.length > 0 && activeTab !== "alerts" && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 p-3 bg-red-950/30 border border-red-900/50 rounded-md flex items-center justify-between cursor-pointer hover:bg-red-950/40 transition-colors"
              onClick={() => setActiveTab("alerts")}
              data-testid="alert-banner"
            >
              <div className="flex items-center gap-3">
                <Bell className="w-5 h-5 text-red-400 animate-pulse" />
                <span className="text-red-400 font-medium">
                  {alerts.length} unacknowledged alert{alerts.length > 1 ? "s" : ""}
                </span>
              </div>
              <span className="text-sm text-red-400/70">Click to view</span>
            </motion.div>
          )}
          
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

export default App;
