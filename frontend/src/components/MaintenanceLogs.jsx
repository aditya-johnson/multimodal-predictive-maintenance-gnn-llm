import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  FileText,
  AlertTriangle,
  Tag,
  User,
  Clock,
  Plus,
  Search,
  Filter,
  ChevronDown
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Label } from "./ui/label";
import { ScrollArea } from "./ui/scroll-area";

const SEVERITY_CONFIG = {
  info: { color: "bg-blue-950/30 text-blue-400 border-blue-900/50", label: "Info" },
  warning: { color: "bg-yellow-950/30 text-yellow-400 border-yellow-900/50", label: "Warning" },
  error: { color: "bg-orange-950/30 text-orange-400 border-orange-900/50", label: "Error" },
  critical: { color: "bg-red-950/30 text-red-400 border-red-900/50", label: "Critical" }
};

const LogEntry = ({ log }) => {
  const severityConfig = SEVERITY_CONFIG[log.severity] || SEVERITY_CONFIG.info;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-zinc-900/50 border border-zinc-800/60 rounded-md p-4 hover:border-zinc-700 transition-colors"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Badge className={`${severityConfig.color} border text-xs`}>
            {severityConfig.label}
          </Badge>
          <span className="text-xs text-zinc-500 font-mono">
            {new Date(log.timestamp).toLocaleString()}
          </span>
        </div>
        {log.embedding_similarity > 0 && (
          <Badge variant="outline" className="border-cyan-500/30 text-cyan-400 text-xs">
            Risk: {(log.embedding_similarity * 100).toFixed(0)}%
          </Badge>
        )}
      </div>
      
      <p className="text-zinc-200 text-sm mb-3 leading-relaxed">{log.log_text}</p>
      
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <User className="w-3 h-3" />
          <span>{log.technician}</span>
        </div>
        
        {log.risk_keywords && log.risk_keywords.length > 0 && (
          <div className="flex items-center gap-1 flex-wrap justify-end">
            <Tag className="w-3 h-3 text-yellow-500" />
            {log.risk_keywords.slice(0, 3).map((kw, i) => (
              <span 
                key={i} 
                className="text-xs px-1.5 py-0.5 bg-yellow-950/30 text-yellow-400 rounded"
              >
                {kw}
              </span>
            ))}
            {log.risk_keywords.length > 3 && (
              <span className="text-xs text-zinc-500">+{log.risk_keywords.length - 3}</span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
};

const RiskKeywordCloud = ({ logs }) => {
  const keywordCounts = {};
  logs.forEach(log => {
    (log.risk_keywords || []).forEach(kw => {
      keywordCounts[kw] = (keywordCounts[kw] || 0) + 1;
    });
  });
  
  const sortedKeywords = Object.entries(keywordCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12);

  if (sortedKeywords.length === 0) return null;

  return (
    <Card className="bg-zinc-950/50 border-zinc-800/60">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-zinc-300 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-yellow-400" />
          Risk Keywords Detected
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2">
          {sortedKeywords.map(([keyword, count]) => (
            <Badge 
              key={keyword} 
              variant="outline" 
              className="border-yellow-900/50 text-yellow-400 bg-yellow-950/20"
            >
              {keyword} ({count})
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export const MaintenanceLogs = ({ selectedMachine, machines, API }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [filterSeverity, setFilterSeverity] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  
  const [newLog, setNewLog] = useState({
    log_text: "",
    technician: "",
    severity: "info"
  });

  useEffect(() => {
    const fetchLogs = async () => {
      if (!selectedMachine) return;
      setLoading(true);
      try {
        const response = await axios.get(`${API}/machines/${selectedMachine.id}/maintenance-logs`);
        setLogs(response.data);
      } catch (error) {
        console.error("Error fetching logs:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchLogs();
  }, [selectedMachine, API]);

  const handleCreateLog = async () => {
    if (!selectedMachine || !newLog.log_text || !newLog.technician) {
      toast.error("Please fill in all fields");
      return;
    }
    
    try {
      toast.loading("Creating log entry...", { id: "create-log" });
      const response = await axios.post(`${API}/maintenance-logs`, {
        machine_id: selectedMachine.id,
        ...newLog
      });
      setLogs([response.data, ...logs]);
      setNewLog({ log_text: "", technician: "", severity: "info" });
      setDialogOpen(false);
      toast.success("Log entry created!", { id: "create-log" });
    } catch (error) {
      toast.error("Failed to create log", { id: "create-log" });
    }
  };

  const filteredLogs = logs.filter(log => {
    const matchesSeverity = filterSeverity === "all" || log.severity === filterSeverity;
    const matchesSearch = !searchTerm || 
      log.log_text.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.technician.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.risk_keywords || []).some(kw => kw.toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesSeverity && matchesSearch;
  });

  if (!selectedMachine) {
    return (
      <Card className="bg-zinc-950/50 border-zinc-800/60 p-12 text-center">
        <FileText className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
        <p className="text-zinc-500">Select a machine to view maintenance logs</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6" data-testid="maintenance-logs">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Maintenance Logs</h1>
          <p className="text-zinc-500 mt-1">NLP-analyzed log entries for {selectedMachine.name}</p>
        </div>
        
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-cyan-500 text-black hover:bg-cyan-400" data-testid="add-log-btn">
              <Plus className="w-4 h-4 mr-2" />
              Add Log Entry
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-zinc-900 border-zinc-800">
            <DialogHeader>
              <DialogTitle className="text-zinc-100">New Maintenance Log</DialogTitle>
              <DialogDescription className="text-zinc-500">
                Add a new maintenance log entry. Risk keywords will be automatically extracted.
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-zinc-300">Log Description</Label>
                <Textarea
                  placeholder="Describe the maintenance observation or action..."
                  value={newLog.log_text}
                  onChange={(e) => setNewLog({ ...newLog, log_text: e.target.value })}
                  className="bg-zinc-800/50 border-zinc-700 text-zinc-100 min-h-[100px]"
                  data-testid="log-text-input"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-zinc-300">Technician Name</Label>
                  <Input
                    placeholder="Enter name"
                    value={newLog.technician}
                    onChange={(e) => setNewLog({ ...newLog, technician: e.target.value })}
                    className="bg-zinc-800/50 border-zinc-700 text-zinc-100"
                    data-testid="technician-input"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label className="text-zinc-300">Severity</Label>
                  <Select 
                    value={newLog.severity} 
                    onValueChange={(value) => setNewLog({ ...newLog, severity: value })}
                  >
                    <SelectTrigger className="bg-zinc-800/50 border-zinc-700" data-testid="severity-select">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-zinc-900 border-zinc-800">
                      <SelectItem value="info">Info</SelectItem>
                      <SelectItem value="warning">Warning</SelectItem>
                      <SelectItem value="error">Error</SelectItem>
                      <SelectItem value="critical">Critical</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
            
            <DialogFooter>
              <Button variant="outline" onClick={() => setDialogOpen(false)} className="border-zinc-700">
                Cancel
              </Button>
              <Button onClick={handleCreateLog} className="bg-cyan-500 text-black hover:bg-cyan-400">
                Create Entry
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* NLP Analysis Info */}
      <Card className="bg-zinc-950/50 border-zinc-800/60">
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-violet-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-zinc-100">NLP Embedding Analysis</h3>
              <p className="text-xs text-zinc-500">
                Using SentenceTransformers to extract risk keywords and calculate semantic similarity scores
              </p>
            </div>
            <Badge variant="outline" className="border-zinc-700 text-zinc-400">
              {logs.length} entries
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Risk Keywords Cloud */}
      <RiskKeywordCloud logs={logs} />

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <Input
            placeholder="Search logs, technicians, keywords..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-zinc-900/50 border-zinc-800 text-zinc-100"
            data-testid="search-logs-input"
          />
        </div>
        
        <Select value={filterSeverity} onValueChange={setFilterSeverity}>
          <SelectTrigger className="w-[150px] bg-zinc-900/50 border-zinc-800" data-testid="filter-severity">
            <Filter className="w-4 h-4 mr-2 text-zinc-500" />
            <SelectValue placeholder="Filter" />
          </SelectTrigger>
          <SelectContent className="bg-zinc-900 border-zinc-800">
            <SelectItem value="all">All Severity</SelectItem>
            <SelectItem value="info">Info</SelectItem>
            <SelectItem value="warning">Warning</SelectItem>
            <SelectItem value="error">Error</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Logs List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400" />
        </div>
      ) : filteredLogs.length > 0 ? (
        <ScrollArea className="h-[600px]">
          <div className="space-y-3 pr-4">
            {filteredLogs.map((log, index) => (
              <LogEntry key={log.id || index} log={log} />
            ))}
          </div>
        </ScrollArea>
      ) : (
        <Card className="bg-zinc-950/50 border-zinc-800/60 p-8 text-center">
          <FileText className="w-10 h-10 text-zinc-600 mx-auto mb-3" />
          <p className="text-zinc-500">
            {searchTerm || filterSeverity !== "all" 
              ? "No logs match your filters" 
              : "No maintenance logs yet for this machine"}
          </p>
        </Card>
      )}
    </div>
  );
};
