import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, LogOut, User, Building2, FileDown, ChevronDown } from "lucide-react";

import { AuthPage } from "./components/AuthPage";
import { Sidebar } from "./components/Sidebar";
import { HealthDashboard } from "./components/HealthDashboard";
import { SensorTimeSeries } from "./components/SensorTimeSeries";
import { FailurePrediction } from "./components/FailurePrediction";
import { GraphVisualization } from "./components/GraphVisualization";
import { MaintenanceLogs } from "./components/MaintenanceLogs";
import { AlertsPanel } from "./components/AlertsPanel";
import { OrganizationManager } from "./components/OrganizationManager";
import { ComparisonDashboard } from "./components/ComparisonDashboard";
import { useWebSocket } from "./hooks/useWebSocket";
import { Button } from "./components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";

import "@/App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Configure axios
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
      localStorage.removeItem("org");
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [currentOrg, setCurrentOrg] = useState(null);
  const [userRole, setUserRole] = useState(null);
  const [machines, setMachines] = useState([]);
  const [selectedMachine, setSelectedMachine] = useState(null);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [loading, setLoading] = useState(true);
  const [dashboardSummary, setDashboardSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [downloadingReport, setDownloadingReport] = useState(false);

  // Check for existing session
  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    const storedUser = localStorage.getItem("user");
    const storedOrg = localStorage.getItem("org");
    
    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
      if (storedOrg) {
        const org = JSON.parse(storedOrg);
        setCurrentOrg(org);
        setUserRole(org.role);
      }
    }
    setLoading(false);
  }, []);

  // WebSocket
  const { lastMessage, isConnected } = useWebSocket(
    selectedMachine?.id && token
      ? `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/${selectedMachine.id}`
      : null
  );

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
        toast.error(`${lastMessage.data.alert_type}: ${lastMessage.data.machine_name}`, {
          description: lastMessage.data.message,
          duration: 10000
        });
        fetchAlerts();
      }
    }
  }, [lastMessage, selectedMachine]);

  const fetchMachines = useCallback(async () => {
    if (!token || !currentOrg) return;
    try {
      const response = await axios.get(`${API}/machines`);
      setMachines(response.data);
      if (response.data.length > 0 && !selectedMachine) {
        setSelectedMachine(response.data[0]);
      }
    } catch (error) {
      console.error("Error fetching machines:", error);
    }
  }, [token, currentOrg, selectedMachine]);

  const fetchDashboardSummary = async () => {
    if (!token || !currentOrg) return;
    try {
      const response = await axios.get(`${API}/dashboard/summary`);
      setDashboardSummary(response.data);
    } catch (error) {
      console.error("Error fetching summary:", error);
    }
  };

  const fetchAlerts = async () => {
    if (!token || !currentOrg) return;
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
      toast.success("Demo data loaded!", { id: "seed" });
      await fetchMachines();
      await fetchDashboardSummary();
      await fetchAlerts();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to seed demo data", { id: "seed" });
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

  const downloadReport = async (machineId, days = 30) => {
    setDownloadingReport(true);
    try {
      toast.loading("Generating PDF report...", { id: "report" });
      const response = await axios.get(`${API}/machines/${machineId}/report?days=${days}`, {
        responseType: 'blob'
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      const machine = machines.find(m => m.id === machineId);
      link.setAttribute('download', `${machine?.name || 'Machine'}_Report.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success("Report downloaded!", { id: "report" });
    } catch (error) {
      toast.error("Failed to generate report", { id: "report" });
    } finally {
      setDownloadingReport(false);
    }
  };

  const handleLogin = (userData, accessToken) => {
    setUser(userData);
    setToken(accessToken);
    localStorage.setItem("token", accessToken);
    localStorage.setItem("user", JSON.stringify(userData));
  };

  const handleOrgChange = (org, role) => {
    setCurrentOrg(org);
    setUserRole(role);
    localStorage.setItem("org", JSON.stringify({ ...org, role }));
    setSelectedMachine(null);
    setMachines([]);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    localStorage.removeItem("org");
    setUser(null);
    setToken(null);
    setCurrentOrg(null);
    setUserRole(null);
    setMachines([]);
    setSelectedMachine(null);
    toast.success("Logged out");
  };

  useEffect(() => {
    if (token && currentOrg) {
      const init = async () => {
        setLoading(true);
        await fetchMachines();
        await fetchDashboardSummary();
        await fetchAlerts();
        setLoading(false);
      };
      init();
    }
  }, [token, currentOrg]);

  // Show auth page
  if (!token) {
    return (
      <>
        <Toaster theme="dark" position="top-right" />
        <AuthPage onLogin={handleLogin} API={API} />
      </>
    );
  }

  // Check permissions
  const canManageMachines = ["admin", "operator"].includes(userRole);
  const canRunPredictions = ["admin", "operator"].includes(userRole);
  const canManageAlerts = ["admin", "operator"].includes(userRole);
  const canManageOrg = userRole === "admin";

  const renderContent = () => {
    if (!currentOrg) {
      return (
        <OrganizationManager
          currentOrg={currentOrg}
          onOrgChange={handleOrgChange}
          API={API}
        />
      );
    }

    switch (activeTab) {
      case "dashboard":
        return (
          <HealthDashboard
            machines={machines}
            selectedMachine={selectedMachine}
            setSelectedMachine={setSelectedMachine}
            dashboardSummary={dashboardSummary}
            onRunPrediction={canRunPredictions ? runPrediction : null}
            onDownloadReport={downloadReport}
            API={API}
            isStreaming={isConnected}
            userRole={userRole}
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
            onRunPrediction={canRunPredictions ? runPrediction : null}
            API={API}
            userRole={userRole}
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
            canEdit={canManageMachines}
          />
        );
      case "alerts":
        return (
          <AlertsPanel
            alerts={alerts}
            onAcknowledge={canManageAlerts ? acknowledgeAlert : null}
            API={API}
            userRole={userRole}
          />
        );
      case "organization":
        return (
          <OrganizationManager
            currentOrg={currentOrg}
            onOrgChange={handleOrgChange}
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
          style: { background: '#18181b', border: '1px solid #27272a', color: '#e4e4e7' }
        }}
      />
      
      <Sidebar 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        machines={machines}
        selectedMachine={selectedMachine}
        setSelectedMachine={setSelectedMachine}
        onSeedDemo={canManageMachines ? seedDemoData : null}
        loading={loading}
        alertCount={alerts.length}
        hasOrg={!!currentOrg}
        userRole={userRole}
      />
      
      <main className="flex-1 overflow-y-auto flex flex-col">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-zinc-800/60 bg-zinc-900/30">
          <div className="flex items-center gap-4">
            {currentOrg && (
              <div className="flex items-center gap-2 text-sm text-zinc-400">
                <Building2 className="w-4 h-4" />
                <span>{currentOrg.name}</span>
                {userRole && (
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    userRole === "admin" ? "bg-yellow-950/30 text-yellow-400" :
                    userRole === "operator" ? "bg-cyan-950/30 text-cyan-400" :
                    "bg-zinc-800 text-zinc-400"
                  }`}>
                    {userRole}
                  </span>
                )}
              </div>
            )}
            {isConnected && (
              <span className="flex items-center gap-2 text-xs text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Live
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            {/* Download Report */}
            {selectedMachine && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="border-zinc-700" disabled={downloadingReport}>
                    <FileDown className="w-4 h-4 mr-2" />
                    Report
                    <ChevronDown className="w-3 h-3 ml-1" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="bg-zinc-900 border-zinc-800">
                  <DropdownMenuLabel className="text-zinc-400">Download PDF Report</DropdownMenuLabel>
                  <DropdownMenuSeparator className="bg-zinc-800" />
                  <DropdownMenuItem onClick={() => downloadReport(selectedMachine.id, 7)} className="text-zinc-300">
                    Last 7 Days
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => downloadReport(selectedMachine.id, 30)} className="text-zinc-300">
                    Last 30 Days
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => downloadReport(selectedMachine.id, 90)} className="text-zinc-300">
                    Last 90 Days
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            
            {/* User Menu */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="text-zinc-400 hover:text-zinc-100">
                  <User className="w-4 h-4 mr-2" />
                  {user?.name}
                  <ChevronDown className="w-3 h-3 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-zinc-900 border-zinc-800" align="end">
                <DropdownMenuLabel className="text-zinc-400">{user?.email}</DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-zinc-800" />
                <DropdownMenuItem onClick={() => setActiveTab("organization")} className="text-zinc-300">
                  <Building2 className="w-4 h-4 mr-2" />
                  Organizations
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-zinc-800" />
                <DropdownMenuItem onClick={handleLogout} className="text-red-400">
                  <LogOut className="w-4 h-4 mr-2" />
                  Logout
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6 md:p-8 overflow-y-auto">
          {/* Alert Banner */}
          {alerts.length > 0 && activeTab !== "alerts" && currentOrg && (
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
