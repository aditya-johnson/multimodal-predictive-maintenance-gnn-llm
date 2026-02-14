import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  Legend
} from "recharts";
import {
  Thermometer,
  Wind,
  Gauge,
  Zap,
  Clock
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Badge } from "./ui/badge";

const SENSOR_CONFIG = {
  temperature: { color: "#ef4444", icon: Thermometer, unit: "°C", label: "Temperature" },
  vibration: { color: "#facc15", icon: Wind, unit: "mm/s", label: "Vibration" },
  pressure: { color: "#3b82f6", icon: Gauge, unit: "PSI", label: "Pressure" },
  rpm: { color: "#10b981", icon: Zap, unit: "rpm", label: "RPM" }
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  
  return (
    <div className="bg-zinc-900/95 border border-cyan-500/30 backdrop-blur-md rounded-sm p-3 shadow-lg">
      <p className="text-xs text-zinc-500 mb-2 font-mono">
        {new Date(label).toLocaleString()}
      </p>
      {payload.map((entry, index) => (
        <div key={index} className="flex items-center gap-2 text-sm">
          <div 
            className="w-2 h-2 rounded-full" 
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-zinc-400">{entry.name}:</span>
          <span className="font-mono text-zinc-100">
            {entry.value?.toFixed(2)} {SENSOR_CONFIG[entry.dataKey]?.unit}
          </span>
        </div>
      ))}
    </div>
  );
};

export const SensorTimeSeries = ({ selectedMachine, API }) => {
  const [readings, setReadings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedSensor, setSelectedSensor] = useState("temperature");
  const [timeRange, setTimeRange] = useState([0, 100]);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    const fetchReadings = async () => {
      if (!selectedMachine) return;
      setLoading(true);
      try {
        const response = await axios.get(`${API}/machines/${selectedMachine.id}/readings?limit=500`);
        setReadings(response.data);
      } catch (error) {
        console.error("Error fetching readings:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchReadings();
  }, [selectedMachine, API]);

  const filteredData = readings.slice(
    Math.floor((readings.length * timeRange[0]) / 100),
    Math.floor((readings.length * timeRange[1]) / 100)
  );

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit' });
  };

  const stats = {
    min: Math.min(...filteredData.map(d => d[selectedSensor] || 0)),
    max: Math.max(...filteredData.map(d => d[selectedSensor] || 0)),
    avg: filteredData.reduce((sum, d) => sum + (d[selectedSensor] || 0), 0) / filteredData.length || 0
  };

  const SensorIcon = SENSOR_CONFIG[selectedSensor]?.icon || Thermometer;

  if (!selectedMachine) {
    return (
      <Card className="bg-zinc-950/50 border-zinc-800/60 p-12 text-center">
        <p className="text-zinc-500">Select a machine to view sensor data</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6" data-testid="sensor-timeseries">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Sensor Time-Series</h1>
          <p className="text-zinc-500 mt-1">Historical sensor data for {selectedMachine.name}</p>
        </div>
        <Badge variant="outline" className="border-zinc-700 text-zinc-400">
          <Clock className="w-3 h-3 mr-1" />
          {readings.length} readings
        </Badge>
      </div>

      {/* Controls */}
      <Card className="bg-zinc-950/50 border-zinc-800/60">
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-6">
            <div className="flex items-center gap-3">
              <label className="text-sm text-zinc-500">Sensor:</label>
              <Select value={selectedSensor} onValueChange={setSelectedSensor}>
                <SelectTrigger className="w-[180px] bg-zinc-900/50 border-zinc-800" data-testid="sensor-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-800">
                  {Object.entries(SENSOR_CONFIG).map(([key, config]) => {
                    const Icon = config.icon;
                    return (
                      <SelectItem key={key} value={key} className="text-zinc-100">
                        <div className="flex items-center gap-2">
                          <Icon className="w-4 h-4" style={{ color: config.color }} />
                          {config.label}
                        </div>
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex-1 min-w-[200px]">
              <label className="text-sm text-zinc-500 mb-2 block">Time Range:</label>
              <Slider
                value={timeRange}
                onValueChange={setTimeRange}
                min={0}
                max={100}
                step={1}
                className="w-full"
                data-testid="time-range-slider"
              />
              <div className="flex justify-between text-xs text-zinc-600 mt-1">
                <span>{timeRange[0]}%</span>
                <span>{timeRange[1]}%</span>
              </div>
            </div>

            <button
              onClick={() => setShowAll(!showAll)}
              className={`px-4 py-2 text-sm rounded-sm transition-colors ${
                showAll 
                  ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30" 
                  : "bg-zinc-800 text-zinc-400 border border-zinc-700 hover:bg-zinc-700"
              }`}
              data-testid="show-all-toggle"
            >
              {showAll ? "Single Sensor" : "Show All"}
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-zinc-950/50 border-zinc-800/60">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Minimum</p>
            <p className="text-2xl font-mono font-bold text-blue-400">
              {stats.min.toFixed(2)}
            </p>
            <p className="text-xs text-zinc-600">{SENSOR_CONFIG[selectedSensor]?.unit}</p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-950/50 border-zinc-800/60">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Average</p>
            <p className="text-2xl font-mono font-bold text-cyan-400">
              {stats.avg.toFixed(2)}
            </p>
            <p className="text-xs text-zinc-600">{SENSOR_CONFIG[selectedSensor]?.unit}</p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-950/50 border-zinc-800/60">
          <CardContent className="p-4 text-center">
            <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Maximum</p>
            <p className="text-2xl font-mono font-bold text-red-400">
              {stats.max.toFixed(2)}
            </p>
            <p className="text-xs text-zinc-600">{SENSOR_CONFIG[selectedSensor]?.unit}</p>
          </CardContent>
        </Card>
      </div>

      {/* Main Chart */}
      <Card className="bg-zinc-950/50 border-zinc-800/60">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-zinc-100">
            <SensorIcon className="w-5 h-5" style={{ color: SENSOR_CONFIG[selectedSensor]?.color }} />
            {showAll ? "All Sensors" : SENSOR_CONFIG[selectedSensor]?.label} Over Time
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="h-[400px] flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400" />
            </div>
          ) : filteredData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              {showAll ? (
                <LineChart data={filteredData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTime}
                    stroke="#52525b"
                    tick={{ fill: '#71717a', fontSize: 11 }}
                  />
                  <YAxis stroke="#52525b" tick={{ fill: '#71717a', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  {Object.entries(SENSOR_CONFIG).map(([key, config]) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      name={config.label}
                      stroke={config.color}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4, fill: config.color }}
                    />
                  ))}
                </LineChart>
              ) : (
                <AreaChart data={filteredData}>
                  <defs>
                    <linearGradient id={`gradient-${selectedSensor}`} x1="0" y1="0" x2="0" y2="1">
                      <stop 
                        offset="5%" 
                        stopColor={SENSOR_CONFIG[selectedSensor]?.color} 
                        stopOpacity={0.3}
                      />
                      <stop 
                        offset="95%" 
                        stopColor={SENSOR_CONFIG[selectedSensor]?.color} 
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatTime}
                    stroke="#52525b"
                    tick={{ fill: '#71717a', fontSize: 11 }}
                  />
                  <YAxis stroke="#52525b" tick={{ fill: '#71717a', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey={selectedSensor}
                    stroke={SENSOR_CONFIG[selectedSensor]?.color}
                    strokeWidth={2}
                    fill={`url(#gradient-${selectedSensor})`}
                    activeDot={{ r: 4, fill: SENSOR_CONFIG[selectedSensor]?.color }}
                  />
                </AreaChart>
              )}
            </ResponsiveContainer>
          ) : (
            <div className="h-[400px] flex items-center justify-center text-zinc-500">
              No data available
            </div>
          )}
        </CardContent>
      </Card>

      {/* Individual Sensor Charts */}
      {!showAll && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(SENSOR_CONFIG).filter(([key]) => key !== selectedSensor).map(([key, config]) => {
            const Icon = config.icon;
            return (
              <Card key={key} className="bg-zinc-950/50 border-zinc-800/60">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm text-zinc-300">
                    <Icon className="w-4 h-4" style={{ color: config.color }} />
                    {config.label}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={150}>
                    <AreaChart data={filteredData}>
                      <defs>
                        <linearGradient id={`gradient-small-${key}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={config.color} stopOpacity={0.2} />
                          <stop offset="95%" stopColor={config.color} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <Area
                        type="monotone"
                        dataKey={key}
                        stroke={config.color}
                        strokeWidth={1.5}
                        fill={`url(#gradient-small-${key})`}
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};
