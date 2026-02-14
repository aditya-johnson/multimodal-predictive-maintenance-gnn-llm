import { useState, useEffect, useCallback } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle,
  Gauge,
  LineChart,
  Network,
  FileText,
  Settings,
  Upload,
  RefreshCw,
  Cpu,
  Thermometer,
  Wind,
  Zap,
  Clock,
  TrendingUp
} from "lucide-react";

import { Sidebar } from "./components/Sidebar";
import { HealthDashboard } from "./components/HealthDashboard";
import { SensorTimeSeries } from "./components/SensorTimeSeries";
import { FailurePrediction } from "./components/FailurePrediction";
import { GraphVisualization } from "./components/GraphVisualization";
import { MaintenanceLogs } from "./components/MaintenanceLogs";

import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [machines, setMachines] = useState([]);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [loading, setLoading] = useState(true);
  const [dashboardSummary, setDashboardSummary] = useState(null);

  const fetchMachines = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/machines`);
      setMachines(response.data);
      if (response.data.length > 0 && !selectedMachine) {
        setSelectedMachine(response.data[0]);
      }
    } catch (error) {
      console.error("Error fetching machines:", error);
    }
  }, [selectedMachine]);

  const fetchDashboardSummary = async () => {
    try {
      const response = await axios.get(`${API}/dashboard/summary`);
      setDashboardSummary(response.data);
    } catch (error) {
      console.error("Error fetching summary:", error);
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
    } catch (error) {
      toast.error("Failed to seed demo data", { id: "seed" });
      console.error("Error seeding data:", error);
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async (machineId) => {
    try {
      toast.loading("Running prediction...", { id: "predict" });
      await axios.post(`${API}/machines/${machineId}/predict`);
      toast.success("Prediction complete!", { id: "predict" });
      await fetchMachines();
      await fetchDashboardSummary();
    } catch (error) {
      toast.error("Prediction failed", { id: "predict" });
    }
  };

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await fetchMachines();
      await fetchDashboardSummary();
      setLoading(false);
    };
    init();
  }, []);

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
          />
        );
      case "sensors":
        return (
          <SensorTimeSeries
            selectedMachine={selectedMachine}
            API={API}
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
      />
      
      <main className="flex-1 overflow-y-auto p-6 md:p-8">
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
      </main>
    </div>
  );
}

export default App;
