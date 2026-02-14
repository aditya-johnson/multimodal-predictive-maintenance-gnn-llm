import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import ForceGraph2D from "react-force-graph-2d";
import { motion } from "framer-motion";
import {
  Network,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Info,
  RefreshCw
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";

const SENSOR_COLORS = {
  temperature: "#ef4444",
  pressure: "#3b82f6",
  vibration: "#facc15",
  rpm: "#10b981",
  voltage: "#8b5cf6",
  current: "#ec4899"
};

const SENSOR_LABELS = {
  temperature: "Temperature",
  pressure: "Pressure",
  vibration: "Vibration",
  rpm: "RPM",
  voltage: "Voltage",
  current: "Current"
};

export const GraphVisualization = ({ selectedMachine, API }) => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const graphRef = useRef();

  const fetchGraphData = useCallback(async () => {
    if (!selectedMachine) return;
    setLoading(true);
    try {
      const response = await axios.get(`${API}/machines/${selectedMachine.id}/sensor-graph`);
      // Transform data for force-graph
      const transformedData = {
        nodes: response.data.nodes.map(n => ({
          ...n,
          color: SENSOR_COLORS[n.id] || "#00f0ff",
          val: Math.max(10, n.value || 20)
        })),
        links: response.data.links.map(l => ({
          ...l,
          color: `rgba(0, 240, 255, ${Math.min(l.weight || 0.5, 1)})`
        }))
      };
      setGraphData(transformedData);
    } catch (error) {
      console.error("Error fetching graph data:", error);
    } finally {
      setLoading(false);
    }
  }, [selectedMachine, API]);

  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  const handleNodeClick = (node) => {
    setSelectedNode(node);
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 1000);
      graphRef.current.zoom(2, 1000);
    }
  };

  const handleZoomIn = () => {
    if (graphRef.current) {
      const currentZoom = graphRef.current.zoom();
      graphRef.current.zoom(currentZoom * 1.5, 300);
    }
  };

  const handleZoomOut = () => {
    if (graphRef.current) {
      const currentZoom = graphRef.current.zoom();
      graphRef.current.zoom(currentZoom / 1.5, 300);
    }
  };

  const handleCenter = () => {
    if (graphRef.current) {
      graphRef.current.centerAt(0, 0, 1000);
      graphRef.current.zoom(1, 1000);
    }
  };

  const nodeCanvasObject = useCallback((node, ctx, globalScale) => {
    const label = SENSOR_LABELS[node.id] || node.id;
    const fontSize = 12 / globalScale;
    const nodeRadius = Math.sqrt(node.val) * 2;

    // Draw glow
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius + 4, 0, 2 * Math.PI);
    ctx.fillStyle = `${node.color}20`;
    ctx.fill();

    // Draw node
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI);
    ctx.fillStyle = node.color;
    ctx.fill();

    // Draw border
    ctx.strokeStyle = "#ffffff30";
    ctx.lineWidth = 1 / globalScale;
    ctx.stroke();

    // Draw label
    ctx.font = `${fontSize}px JetBrains Mono, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#e4e4e7";
    ctx.fillText(label, node.x, node.y + nodeRadius + fontSize + 2);
  }, []);

  const linkCanvasObject = useCallback((link, ctx) => {
    const start = link.source;
    const end = link.target;

    // Draw link with gradient
    const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
    gradient.addColorStop(0, `${SENSOR_COLORS[start.id] || "#00f0ff"}80`);
    gradient.addColorStop(1, `${SENSOR_COLORS[end.id] || "#00f0ff"}80`);

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = (link.weight || 0.5) * 3;
    ctx.stroke();
  }, []);

  if (!selectedMachine) {
    return (
      <Card className="bg-zinc-950/50 border-zinc-800/60 p-12 text-center">
        <Network className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
        <p className="text-zinc-500">Select a machine to view sensor graph</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6" data-testid="graph-visualization">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Sensor Dependency Graph</h1>
          <p className="text-zinc-500 mt-1">GNN correlation modeling for {selectedMachine.name}</p>
        </div>
        <Button
          onClick={fetchGraphData}
          disabled={loading}
          variant="outline"
          className="border-zinc-700"
          data-testid="refresh-graph-btn"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Graph */}
        <Card className="lg:col-span-3 bg-zinc-950/50 border-zinc-800/60 overflow-hidden">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-zinc-100 flex items-center gap-2">
              <Network className="w-5 h-5 text-cyan-400" />
              Correlation Network
            </CardTitle>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="icon" onClick={handleZoomIn} className="h-8 w-8 text-zinc-400">
                <ZoomIn className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon" onClick={handleZoomOut} className="h-8 w-8 text-zinc-400">
                <ZoomOut className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="icon" onClick={handleCenter} className="h-8 w-8 text-zinc-400">
                <Maximize2 className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-[500px] bg-zinc-900/30">
              {loading ? (
                <div className="h-full flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400" />
                </div>
              ) : (
                <ForceGraph2D
                  ref={graphRef}
                  graphData={graphData}
                  nodeCanvasObject={nodeCanvasObject}
                  linkCanvasObject={linkCanvasObject}
                  onNodeClick={handleNodeClick}
                  backgroundColor="#09090b"
                  linkDirectionalParticles={2}
                  linkDirectionalParticleWidth={2}
                  linkDirectionalParticleColor={() => "#00f0ff"}
                  cooldownTicks={100}
                  d3AlphaDecay={0.02}
                  d3VelocityDecay={0.3}
                />
              )}
            </div>
          </CardContent>
        </Card>

        {/* Info Panel */}
        <div className="space-y-4">
          {/* Legend */}
          <Card className="bg-zinc-950/50 border-zinc-800/60">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">Sensor Legend</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {Object.entries(SENSOR_COLORS).map(([sensor, color]) => (
                <div key={sensor} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-sm text-zinc-400">{SENSOR_LABELS[sensor]}</span>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Selected Node Info */}
          {selectedNode && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card className="bg-zinc-950/50 border-cyan-500/30">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-cyan-400 flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    Selected Node
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wider">Sensor</p>
                    <p className="text-lg font-semibold text-zinc-100">
                      {SENSOR_LABELS[selectedNode.id] || selectedNode.id}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wider">Variance Score</p>
                    <p className="text-2xl font-mono font-bold text-cyan-400">
                      {selectedNode.value?.toFixed(1) || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Group</p>
                    <Badge variant="outline" className="border-zinc-700 text-zinc-400">
                      Cluster {selectedNode.group}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Graph Stats */}
          <Card className="bg-zinc-950/50 border-zinc-800/60">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">Graph Statistics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-zinc-500">Nodes</span>
                <span className="font-mono text-zinc-100">{graphData.nodes.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-zinc-500">Edges</span>
                <span className="font-mono text-zinc-100">{graphData.links.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-zinc-500">Avg. Correlation</span>
                <span className="font-mono text-cyan-400">
                  {graphData.links.length > 0 
                    ? (graphData.links.reduce((sum, l) => sum + (l.weight || 0), 0) / graphData.links.length).toFixed(3)
                    : "—"
                  }
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Explanation */}
          <Card className="bg-zinc-900/30 border-zinc-800/40">
            <CardContent className="p-4">
              <p className="text-xs text-zinc-500 leading-relaxed">
                <strong className="text-zinc-400">How it works:</strong> The graph shows correlations 
                between sensors computed from historical data. Thicker edges indicate stronger 
                correlations. Node size represents variance (anomaly indicator). The GNN uses 
                this structure to learn sensor dependencies for better failure prediction.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
