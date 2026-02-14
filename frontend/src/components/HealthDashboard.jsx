import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Thermometer,
  Wind,
  Gauge,
  Zap,
  TrendingUp,
  Clock,
  RefreshCw,
  MapPin
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";

const HealthGauge = ({ value, size = 180 }) => {
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * Math.PI * 1.5; // 270 degrees
  const offset = circumference - (value / 100) * circumference;
  
  const getColor = (val) => {
    if (val >= 70) return "#10b981";
    if (val >= 40) return "#facc15";
    return "#ef4444";
  };

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-135">
        {/* Background arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#27272a"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeLinecap="round"
        />
        {/* Value arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor(value)}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{
            filter: `drop-shadow(0 0 8px ${getColor(value)}40)`
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-mono font-bold text-zinc-100">
          {value.toFixed(0)}
        </span>
        <span className="text-sm text-zinc-500 uppercase tracking-wider">Health</span>
      </div>
    </div>
  );
};

const StatusBadge = ({ level }) => {
  const config = {
    healthy: { color: "bg-emerald-950/30 text-emerald-400 border-emerald-900/50", icon: CheckCircle2, label: "Healthy" },
    warning: { color: "bg-yellow-950/30 text-yellow-400 border-yellow-900/50", icon: AlertTriangle, label: "Warning" },
    critical: { color: "bg-red-950/30 text-red-400 border-red-900/50", icon: XCircle, label: "Critical" }
  };
  
  const { color, icon: Icon, label } = config[level] || config.healthy;
  
  return (
    <Badge className={`${color} border px-3 py-1 text-xs font-mono uppercase tracking-wider`}>
      <Icon className="w-3 h-3 mr-1.5" />
      {label}
    </Badge>
  );
};

const SensorCard = ({ icon: Icon, label, value, unit, trend, status }) => {
  const statusColors = {
    normal: "text-emerald-400",
    warning: "text-yellow-400",
    critical: "text-red-400"
  };

  return (
    <div className="bg-zinc-900/50 border border-zinc-800/60 rounded-md p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
        </div>
        {trend && (
          <TrendingUp className={`w-4 h-4 ${trend > 0 ? "text-red-400" : "text-emerald-400"} ${trend > 0 ? "" : "rotate-180"}`} />
        )}
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-mono font-semibold ${statusColors[status] || "text-zinc-100"}`}>
          {value?.toFixed(1) || "—"}
        </span>
        <span className="text-sm text-zinc-500">{unit}</span>
      </div>
    </div>
  );
};

const SummaryCard = ({ icon: Icon, label, value, color }) => (
  <div className="flex items-center gap-3 p-4 bg-zinc-900/30 rounded-md border border-zinc-800/40">
    <div className={`w-10 h-10 rounded-md flex items-center justify-center ${color}`}>
      <Icon className="w-5 h-5" />
    </div>
    <div>
      <p className="text-2xl font-mono font-bold text-zinc-100">{value}</p>
      <p className="text-xs text-zinc-500 uppercase tracking-wider">{label}</p>
    </div>
  </div>
);

export const HealthDashboard = ({ 
  machines, 
  selectedMachine, 
  setSelectedMachine, 
  dashboardSummary,
  onRunPrediction,
  API 
}) => {
  const [latestReadings, setLatestReadings] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchLatestReadings = async () => {
      if (!selectedMachine) return;
      try {
        const response = await axios.get(`${API}/machines/${selectedMachine.id}/readings?limit=24`);
        if (response.data.length > 0) {
          setLatestReadings(response.data[response.data.length - 1]);
        }
      } catch (error) {
        console.error("Error fetching readings:", error);
      }
    };
    fetchLatestReadings();
  }, [selectedMachine, API]);

  const handleRunPrediction = async () => {
    if (!selectedMachine) return;
    setLoading(true);
    await onRunPrediction(selectedMachine.id);
    setLoading(false);
  };

  const getSensorStatus = (sensor, value) => {
    const thresholds = {
      temperature: { warning: 60, critical: 75 },
      vibration: { warning: 2, critical: 3.5 },
      pressure: { warning: 85, critical: 75 },
      rpm: { warning: 2600, critical: 2400 }
    };
    
    const t = thresholds[sensor];
    if (!t) return "normal";
    
    if (sensor === "pressure" || sensor === "rpm") {
      if (value < t.critical) return "critical";
      if (value < t.warning) return "warning";
    } else {
      if (value > t.critical) return "critical";
      if (value > t.warning) return "warning";
    }
    return "normal";
  };

  return (
    <div className="space-y-6" data-testid="health-dashboard">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Equipment Health</h1>
          <p className="text-zinc-500 mt-1">Real-time monitoring and predictive analytics</p>
        </div>
        <Button
          onClick={handleRunPrediction}
          disabled={!selectedMachine || loading}
          className="bg-cyan-500 text-black hover:bg-cyan-400 font-medium"
          data-testid="run-prediction-btn"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Run Prediction
        </Button>
      </div>

      {/* Summary Cards */}
      {dashboardSummary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <SummaryCard 
            icon={Activity} 
            label="Total Equipment" 
            value={dashboardSummary.total_machines}
            color="bg-cyan-500/10 text-cyan-400"
          />
          <SummaryCard 
            icon={CheckCircle2} 
            label="Healthy" 
            value={dashboardSummary.healthy}
            color="bg-emerald-500/10 text-emerald-400"
          />
          <SummaryCard 
            icon={AlertTriangle} 
            label="Warning" 
            value={dashboardSummary.warning}
            color="bg-yellow-500/10 text-yellow-400"
          />
          <SummaryCard 
            icon={XCircle} 
            label="Critical" 
            value={dashboardSummary.critical}
            color="bg-red-500/10 text-red-400"
          />
        </div>
      )}

      {selectedMachine ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Main Health Card */}
          <Card className="lg:col-span-5 bg-zinc-950/50 border-zinc-800/60" data-testid="health-card">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-xl text-zinc-100">{selectedMachine.name}</CardTitle>
                  <p className="text-sm text-zinc-500 mt-1">{selectedMachine.machine_type}</p>
                </div>
                <StatusBadge level={selectedMachine.risk_level} />
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center py-6">
                <HealthGauge value={selectedMachine.health_score || 0} />
                
                <div className="w-full mt-6 space-y-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-zinc-500">Failure Probability</span>
                    <span className="font-mono text-zinc-100">{selectedMachine.failure_probability?.toFixed(1) || 0}%</span>
                  </div>
                  <Progress 
                    value={selectedMachine.failure_probability || 0} 
                    className="h-2 bg-zinc-800"
                  />
                  
                  <div className="flex items-center gap-2 text-sm text-zinc-500 pt-2">
                    <MapPin className="w-4 h-4" />
                    <span>{selectedMachine.location}</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Sensor Readings */}
          <div className="lg:col-span-7 space-y-4">
            <h3 className="text-lg font-semibold text-zinc-100">Live Sensor Readings</h3>
            <div className="grid grid-cols-2 gap-4">
              <SensorCard
                icon={Thermometer}
                label="Temperature"
                value={latestReadings?.temperature}
                unit="°C"
                status={getSensorStatus("temperature", latestReadings?.temperature)}
              />
              <SensorCard
                icon={Wind}
                label="Vibration"
                value={latestReadings?.vibration}
                unit="mm/s"
                status={getSensorStatus("vibration", latestReadings?.vibration)}
              />
              <SensorCard
                icon={Gauge}
                label="Pressure"
                value={latestReadings?.pressure}
                unit="PSI"
                status={getSensorStatus("pressure", latestReadings?.pressure)}
              />
              <SensorCard
                icon={Zap}
                label="RPM"
                value={latestReadings?.rpm}
                unit="rpm"
                status={getSensorStatus("rpm", latestReadings?.rpm)}
              />
            </div>
            
            {latestReadings && (
              <div className="flex items-center gap-2 text-sm text-zinc-500 mt-4">
                <Clock className="w-4 h-4" />
                <span>Last updated: {new Date(latestReadings.timestamp).toLocaleString()}</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <Card className="bg-zinc-950/50 border-zinc-800/60 p-12 text-center">
          <Activity className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
          <p className="text-zinc-500">Select a machine from the sidebar or load demo data to begin</p>
        </Card>
      )}

      {/* Machine Grid */}
      {machines.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-zinc-100 mb-4">All Equipment</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {machines.map((machine) => (
              <motion.div
                key={machine.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Card 
                  className={`cursor-pointer transition-all bg-zinc-950/50 border-zinc-800/60 hover:border-cyan-500/30 ${
                    selectedMachine?.id === machine.id ? "border-cyan-500/50 shadow-[0_0_15px_rgba(0,240,255,0.1)]" : ""
                  }`}
                  onClick={() => setSelectedMachine(machine)}
                  data-testid={`equipment-card-${machine.id}`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <h4 className="font-semibold text-zinc-100">{machine.name}</h4>
                        <p className="text-sm text-zinc-500">{machine.machine_type}</p>
                      </div>
                      <div className="text-right">
                        <p className={`text-2xl font-mono font-bold ${
                          machine.risk_level === "healthy" ? "text-emerald-400" :
                          machine.risk_level === "warning" ? "text-yellow-400" : "text-red-400"
                        }`}>
                          {machine.health_score?.toFixed(0) || "—"}%
                        </p>
                        <StatusBadge level={machine.risk_level} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
