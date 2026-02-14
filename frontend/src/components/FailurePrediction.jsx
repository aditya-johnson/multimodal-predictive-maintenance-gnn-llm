import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Clock,
  Calendar,
  Target,
  TrendingUp,
  Brain,
  Network,
  FileText,
  RefreshCw,
  ChevronRight
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";

const ScoreGauge = ({ label, value, icon: Icon, color }) => {
  const percentage = Math.round(value * 100);
  
  return (
    <div className="bg-zinc-900/50 border border-zinc-800/60 rounded-md p-4">
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`w-4 h-4 ${color}`} />
        <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-end gap-2 mb-2">
        <span className="text-3xl font-mono font-bold text-zinc-100">{percentage}</span>
        <span className="text-sm text-zinc-500 mb-1">%</span>
      </div>
      <Progress value={percentage} className="h-1.5 bg-zinc-800" />
    </div>
  );
};

const PredictionCard = ({ prediction }) => {
  const getRiskColor = (score) => {
    if (score < 0.3) return "text-emerald-400";
    if (score < 0.6) return "text-yellow-400";
    return "text-red-400";
  };

  const getRiskBg = (score) => {
    if (score < 0.3) return "bg-emerald-950/30 border-emerald-900/50";
    if (score < 0.6) return "bg-yellow-950/30 border-yellow-900/50";
    return "bg-red-950/30 border-red-900/50";
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`border rounded-md p-6 ${getRiskBg(prediction.fusion_score)}`}
    >
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* RUL */}
        <div className="text-center">
          <Clock className={`w-8 h-8 mx-auto mb-2 ${getRiskColor(prediction.fusion_score)}`} />
          <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Remaining Useful Life</p>
          <p className={`text-4xl font-mono font-bold ${getRiskColor(prediction.fusion_score)}`}>
            {prediction.remaining_useful_life_days?.toFixed(0)}
          </p>
          <p className="text-sm text-zinc-500">days</p>
        </div>

        {/* Predicted Date */}
        <div className="text-center">
          <Calendar className="w-8 h-8 mx-auto mb-2 text-cyan-400" />
          <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Predicted Failure</p>
          <p className="text-lg font-mono font-semibold text-zinc-100">
            {new Date(prediction.predicted_failure_date).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric'
            })}
          </p>
        </div>

        {/* Confidence */}
        <div className="text-center">
          <Target className="w-8 h-8 mx-auto mb-2 text-violet-400" />
          <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Confidence Score</p>
          <p className="text-4xl font-mono font-bold text-violet-400">
            {(prediction.confidence_score * 100).toFixed(0)}%
          </p>
        </div>

        {/* Failure Type */}
        <div className="text-center">
          <AlertTriangle className={`w-8 h-8 mx-auto mb-2 ${getRiskColor(prediction.fusion_score)}`} />
          <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Failure Type</p>
          <Badge className={`${getRiskBg(prediction.fusion_score)} text-sm font-medium`}>
            {prediction.failure_type}
          </Badge>
        </div>
      </div>
    </motion.div>
  );
};

export const FailurePrediction = ({ selectedMachine, onRunPrediction, API }) => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [latestPrediction, setLatestPrediction] = useState(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      if (!selectedMachine) return;
      setLoading(true);
      try {
        const response = await axios.get(`${API}/machines/${selectedMachine.id}/predictions`);
        setPredictions(response.data);
        if (response.data.length > 0) {
          setLatestPrediction(response.data[0]);
        }
      } catch (error) {
        console.error("Error fetching predictions:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchPredictions();
  }, [selectedMachine, API]);

  const handleRunPrediction = async () => {
    if (!selectedMachine) return;
    setLoading(true);
    await onRunPrediction(selectedMachine.id);
    // Refetch predictions
    try {
      const response = await axios.get(`${API}/machines/${selectedMachine.id}/predictions`);
      setPredictions(response.data);
      if (response.data.length > 0) {
        setLatestPrediction(response.data[0]);
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
    }
    setLoading(false);
  };

  if (!selectedMachine) {
    return (
      <Card className="bg-zinc-950/50 border-zinc-800/60 p-12 text-center">
        <Brain className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
        <p className="text-zinc-500">Select a machine to view failure predictions</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6" data-testid="failure-prediction">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Failure Prediction</h1>
          <p className="text-zinc-500 mt-1">Multimodal AI prediction for {selectedMachine.name}</p>
        </div>
        <Button
          onClick={handleRunPrediction}
          disabled={loading}
          className="bg-cyan-500 text-black hover:bg-cyan-400 font-medium"
          data-testid="run-prediction-btn"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Run Prediction
        </Button>
      </div>

      {/* Model Explanation */}
      <Card className="bg-zinc-950/50 border-zinc-800/60">
        <CardContent className="p-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 rounded-lg bg-cyan-500/10 flex items-center justify-center">
              <Brain className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-zinc-100">Multimodal Fusion Model</h3>
              <p className="text-sm text-zinc-500">GNN + NLP embeddings for enhanced prediction accuracy</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 p-3 bg-zinc-900/50 rounded-md border border-zinc-800/60">
              <Network className="w-5 h-5 text-cyan-400" />
              <div>
                <p className="text-sm font-medium text-zinc-300">Graph Neural Network</p>
                <p className="text-xs text-zinc-500">Sensor dependency modeling</p>
              </div>
              <ChevronRight className="w-4 h-4 text-zinc-600 ml-auto" />
            </div>
            <div className="flex items-center gap-3 p-3 bg-zinc-900/50 rounded-md border border-zinc-800/60">
              <FileText className="w-5 h-5 text-violet-400" />
              <div>
                <p className="text-sm font-medium text-zinc-300">NLP Embeddings</p>
                <p className="text-xs text-zinc-500">Maintenance log analysis</p>
              </div>
              <ChevronRight className="w-4 h-4 text-zinc-600 ml-auto" />
            </div>
            <div className="flex items-center gap-3 p-3 bg-zinc-900/50 rounded-md border border-zinc-800/60">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              <div>
                <p className="text-sm font-medium text-zinc-300">Fusion Layer</p>
                <p className="text-xs text-zinc-500">Combined prediction</p>
              </div>
              <ChevronRight className="w-4 h-4 text-zinc-600 ml-auto" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Latest Prediction */}
      {latestPrediction ? (
        <>
          <h3 className="text-lg font-semibold text-zinc-100">Latest Prediction</h3>
          <PredictionCard prediction={latestPrediction} />

          {/* Score Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <ScoreGauge
              label="GNN Score"
              value={latestPrediction.gnn_score}
              icon={Network}
              color="text-cyan-400"
            />
            <ScoreGauge
              label="NLP Score"
              value={latestPrediction.nlp_score}
              icon={FileText}
              color="text-violet-400"
            />
            <ScoreGauge
              label="Fusion Score"
              value={latestPrediction.fusion_score}
              icon={Brain}
              color="text-emerald-400"
            />
          </div>
        </>
      ) : (
        <Card className="bg-zinc-950/50 border-zinc-800/60 p-8 text-center">
          <Target className="w-10 h-10 text-zinc-600 mx-auto mb-3" />
          <p className="text-zinc-500 mb-4">No predictions yet for this machine</p>
          <Button
            onClick={handleRunPrediction}
            disabled={loading}
            variant="outline"
            className="border-zinc-700"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Generate First Prediction
          </Button>
        </Card>
      )}

      {/* Prediction History */}
      {predictions.length > 1 && (
        <Card className="bg-zinc-950/50 border-zinc-800/60">
          <CardHeader>
            <CardTitle className="text-zinc-100">Prediction History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {predictions.slice(1, 6).map((pred, index) => (
                <div
                  key={pred.id}
                  className="flex items-center justify-between p-3 bg-zinc-900/30 rounded-md border border-zinc-800/40"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-zinc-600 font-mono w-6">#{index + 2}</span>
                    <div>
                      <p className="text-sm text-zinc-300">
                        RUL: <span className="font-mono">{pred.remaining_useful_life_days?.toFixed(0)} days</span>
                      </p>
                      <p className="text-xs text-zinc-500">
                        {new Date(pred.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <Badge variant="outline" className="border-zinc-700 text-zinc-400 text-xs">
                      {pred.failure_type}
                    </Badge>
                    <span className="text-sm font-mono text-zinc-400">
                      {(pred.confidence_score * 100).toFixed(0)}% conf
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
