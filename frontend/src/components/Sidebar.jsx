import { useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Gauge,
  LineChart,
  Network,
  FileText,
  ChevronLeft,
  ChevronRight,
  Database,
  Cpu,
  AlertTriangle,
  CheckCircle2,
  Bell,
  Building2,
  BarChart3
} from "lucide-react";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Separator } from "./ui/separator";

const navItems = [
  { id: "dashboard", label: "Health Overview", icon: Gauge },
  { id: "sensors", label: "Sensor Data", icon: LineChart },
  { id: "prediction", label: "Failure Prediction", icon: Activity },
  { id: "graph", label: "Sensor Graph", icon: Network },
  { id: "logs", label: "Maintenance Logs", icon: FileText },
  { id: "alerts", label: "Alerts", icon: Bell },
  { id: "comparison", label: "Model Comparison", icon: BarChart3 },
  { id: "organization", label: "Organization", icon: Building2 }
];

const getRiskColor = (level) => {
  switch (level) {
    case "healthy": return "text-emerald-400";
    case "warning": return "text-yellow-400";
    case "critical": return "text-red-400";
    default: return "text-zinc-400";
  }
};

const getRiskIcon = (level) => {
  switch (level) {
    case "healthy": return CheckCircle2;
    case "warning": return AlertTriangle;
    case "critical": return AlertTriangle;
    default: return Cpu;
  }
};

export const Sidebar = ({ 
  activeTab, 
  setActiveTab, 
  machines, 
  selectedMachine, 
  setSelectedMachine,
  onSeedDemo,
  loading,
  alertCount = 0,
  hasOrg = false,
  userRole = null
}) => {
  const [collapsed, setCollapsed] = useState(false);

  // Filter nav items based on organization status
  const visibleNavItems = navItems.filter(item => {
    // Show organization tab always for switching/creating
    if (item.id === "organization") return true;
    // Other tabs require an organization
    return hasOrg;
  });

  return (
    <motion.aside
      className="relative flex flex-col bg-zinc-900/50 border-r border-zinc-800/60 h-full"
      animate={{ width: collapsed ? 72 : 280 }}
      transition={{ duration: 0.2 }}
      data-testid="sidebar"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800/60">
        {!collapsed && (
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-md bg-cyan-500/10 flex items-center justify-center">
              <Activity className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h1 className="font-semibold text-zinc-100 text-sm tracking-tight">PredictMaint</h1>
              <p className="text-xs text-zinc-500">GNN + LLM System</p>
            </div>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-zinc-400 hover:text-zinc-100"
          onClick={() => setCollapsed(!collapsed)}
          data-testid="sidebar-toggle"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="p-2">
        {visibleNavItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          const showBadge = item.id === "alerts" && alertCount > 0;
          return (
            <Button
              key={item.id}
              variant={isActive ? "secondary" : "ghost"}
              className={`w-full justify-start mb-1 relative ${
                isActive 
                  ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20" 
                  : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50"
              } ${collapsed ? "px-3" : "px-4"}`}
              onClick={() => setActiveTab(item.id)}
              data-testid={`nav-${item.id}`}
            >
              <Icon className={`w-4 h-4 ${collapsed ? "" : "mr-3"} ${showBadge ? "text-red-400" : ""}`} />
              {!collapsed && <span className="text-sm">{item.label}</span>}
              {showBadge && (
                <span className={`absolute ${collapsed ? "top-0 right-0" : "right-2"} bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-mono`}>
                  {alertCount > 9 ? "9+" : alertCount}
                </span>
              )}
            </Button>
          );
        })}
      </nav>

      <Separator className="bg-zinc-800/60" />

      {/* Machine List */}
      {!collapsed && (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between px-4 py-3">
            <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
              Equipment
            </span>
            <span className="text-xs text-zinc-600">{machines.length}</span>
          </div>
          
          <ScrollArea className="flex-1 px-2">
            {machines.map((machine) => {
              const RiskIcon = getRiskIcon(machine.risk_level);
              const isSelected = selectedMachine?.id === machine.id;
              return (
                <button
                  key={machine.id}
                  className={`w-full p-3 mb-1 rounded-md text-left transition-all ${
                    isSelected 
                      ? "bg-zinc-800/80 border border-zinc-700" 
                      : "hover:bg-zinc-800/40 border border-transparent"
                  }`}
                  onClick={() => setSelectedMachine(machine)}
                  data-testid={`machine-${machine.id}`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded bg-zinc-800 flex items-center justify-center ${getRiskColor(machine.risk_level)}`}>
                      <RiskIcon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-zinc-200 truncate">
                        {machine.name}
                      </p>
                      <p className="text-xs text-zinc-500 truncate">
                        {machine.machine_type}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className={`text-sm font-mono font-medium ${getRiskColor(machine.risk_level)}`}>
                        {machine.health_score?.toFixed(0) || "—"}%
                      </p>
                    </div>
                  </div>
                </button>
              );
            })}
          </ScrollArea>
        </div>
      )}

      {/* Seed Demo Button */}
      {hasOrg && onSeedDemo && (
        <div className="p-3 border-t border-zinc-800/60">
          <Button
            variant="outline"
            className={`w-full border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 ${
              collapsed ? "px-2" : ""
            }`}
            onClick={onSeedDemo}
            disabled={loading}
            data-testid="seed-demo-btn"
          >
            <Database className={`w-4 h-4 ${collapsed ? "" : "mr-2"}`} />
            {!collapsed && (loading ? "Loading..." : "Load Demo Data")}
          </Button>
        </div>
      )}
    </motion.aside>
  );
};
