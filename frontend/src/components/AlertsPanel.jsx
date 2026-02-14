import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Bell,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Mail,
  Settings,
  Trash2,
  Check
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { Switch } from "./ui/switch";
import { Label } from "./ui/label";
import { Slider } from "./ui/slider";
import { ScrollArea } from "./ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";

const SEVERITY_CONFIG = {
  critical: { 
    color: "bg-red-950/30 text-red-400 border-red-900/50", 
    icon: XCircle,
    bgClass: "bg-red-950/20 border-red-900/30"
  },
  warning: { 
    color: "bg-yellow-950/30 text-yellow-400 border-yellow-900/50", 
    icon: AlertTriangle,
    bgClass: "bg-yellow-950/20 border-yellow-900/30"
  },
  info: { 
    color: "bg-blue-950/30 text-blue-400 border-blue-900/50", 
    icon: Bell,
    bgClass: "bg-blue-950/20 border-blue-900/30"
  }
};

const AlertCard = ({ alert, onAcknowledge }) => {
  const config = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.info;
  const Icon = config.icon;
  
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className={`border rounded-md p-4 ${config.bgClass}`}
    >
      <div className="flex items-start gap-4">
        <div className={`p-2 rounded-md ${config.color}`}>
          <Icon className="w-5 h-5" />
        </div>
        
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-zinc-100">{alert.machine_name}</h4>
              <Badge className={`${config.color} border text-xs`}>
                {alert.severity}
              </Badge>
            </div>
            <span className="text-xs text-zinc-500 font-mono">
              {new Date(alert.timestamp).toLocaleString()}
            </span>
          </div>
          
          <p className="text-sm text-zinc-300 mb-3">{alert.message}</p>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4 text-sm">
              <span className="text-zinc-500">
                Health: <span className={`font-mono ${
                  alert.health_score < 40 ? "text-red-400" : 
                  alert.health_score < 70 ? "text-yellow-400" : "text-emerald-400"
                }`}>{alert.health_score.toFixed(1)}%</span>
              </span>
              <span className="text-zinc-500">
                Failure Risk: <span className="font-mono text-zinc-300">{alert.failure_probability.toFixed(1)}%</span>
              </span>
              {alert.email_sent && (
                <span className="flex items-center gap-1 text-cyan-400 text-xs">
                  <Mail className="w-3 h-3" /> Email sent
                </span>
              )}
            </div>
            
            <Button
              size="sm"
              onClick={() => onAcknowledge(alert.id)}
              className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
              data-testid={`acknowledge-${alert.id}`}
            >
              <Check className="w-4 h-4 mr-1" />
              Acknowledge
            </Button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export const AlertsPanel = ({ alerts, onAcknowledge, API }) => {
  const [settings, setSettings] = useState({
    email_enabled: false,
    email_recipients: [],
    critical_threshold: 40,
    warning_threshold: 70
  });
  const [newEmail, setNewEmail] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, [API]);

  const fetchSettings = async () => {
    try {
      const response = await axios.get(`${API}/alert-settings`);
      setSettings(response.data);
    } catch (error) {
      console.error("Error fetching settings:", error);
    }
  };

  const updateSettings = async (updates) => {
    setSaving(true);
    try {
      const response = await axios.put(`${API}/alert-settings`, updates);
      setSettings(response.data);
      toast.success("Settings updated");
    } catch (error) {
      toast.error("Failed to update settings");
    } finally {
      setSaving(false);
    }
  };

  const addEmail = () => {
    if (newEmail && !settings.email_recipients.includes(newEmail)) {
      const newRecipients = [...settings.email_recipients, newEmail];
      updateSettings({ email_recipients: newRecipients });
      setNewEmail("");
    }
  };

  const removeEmail = (email) => {
    const newRecipients = settings.email_recipients.filter(e => e !== email);
    updateSettings({ email_recipients: newRecipients });
  };

  const criticalAlerts = alerts.filter(a => a.severity === "critical");
  const warningAlerts = alerts.filter(a => a.severity === "warning");

  return (
    <div className="space-y-6" data-testid="alerts-panel">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Alert Center</h1>
          <p className="text-zinc-500 mt-1">Real-time alerts and notification settings</p>
        </div>
        
        <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" className="border-zinc-700" data-testid="alert-settings-btn">
              <Settings className="w-4 h-4 mr-2" />
              Alert Settings
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-zinc-900 border-zinc-800 max-w-lg">
            <DialogHeader>
              <DialogTitle className="text-zinc-100">Alert Settings</DialogTitle>
              <DialogDescription className="text-zinc-500">
                Configure alert thresholds and email notifications
              </DialogDescription>
            </DialogHeader>
            
            <div className="space-y-6 py-4">
              {/* Email Toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-zinc-300">Email Notifications</Label>
                  <p className="text-xs text-zinc-500">Send alerts via email</p>
                </div>
                <Switch
                  checked={settings.email_enabled}
                  onCheckedChange={(checked) => updateSettings({ email_enabled: checked })}
                  data-testid="email-toggle"
                />
              </div>
              
              {/* Email Recipients */}
              {settings.email_enabled && (
                <div className="space-y-3">
                  <Label className="text-zinc-300">Email Recipients</Label>
                  <div className="flex gap-2">
                    <Input
                      type="email"
                      placeholder="Add email address"
                      value={newEmail}
                      onChange={(e) => setNewEmail(e.target.value)}
                      className="bg-zinc-800/50 border-zinc-700"
                      onKeyPress={(e) => e.key === "Enter" && addEmail()}
                      data-testid="email-input"
                    />
                    <Button onClick={addEmail} className="bg-cyan-500 text-black hover:bg-cyan-400">
                      Add
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {settings.email_recipients.map((email) => (
                      <Badge 
                        key={email} 
                        variant="outline" 
                        className="border-zinc-700 text-zinc-300 pr-1"
                      >
                        {email}
                        <button 
                          onClick={() => removeEmail(email)}
                          className="ml-2 hover:text-red-400"
                        >
                          <XCircle className="w-3 h-3" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Thresholds */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-zinc-300">Critical Threshold</Label>
                    <span className="text-sm font-mono text-red-400">{settings.critical_threshold}%</span>
                  </div>
                  <Slider
                    value={[settings.critical_threshold]}
                    onValueChange={([val]) => setSettings(s => ({ ...s, critical_threshold: val }))}
                    onValueCommit={([val]) => updateSettings({ critical_threshold: val })}
                    min={10}
                    max={50}
                    step={5}
                    className="w-full"
                    data-testid="critical-threshold-slider"
                  />
                  <p className="text-xs text-zinc-500 mt-1">Alerts when health drops below this value</p>
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-zinc-300">Warning Threshold</Label>
                    <span className="text-sm font-mono text-yellow-400">{settings.warning_threshold}%</span>
                  </div>
                  <Slider
                    value={[settings.warning_threshold]}
                    onValueChange={([val]) => setSettings(s => ({ ...s, warning_threshold: val }))}
                    onValueCommit={([val]) => updateSettings({ warning_threshold: val })}
                    min={50}
                    max={90}
                    step={5}
                    className="w-full"
                    data-testid="warning-threshold-slider"
                  />
                  <p className="text-xs text-zinc-500 mt-1">Warnings when health drops below this value</p>
                </div>
              </div>
            </div>
            
            <DialogFooter>
              <Button variant="outline" onClick={() => setSettingsOpen(false)} className="border-zinc-700">
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Alert Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-zinc-950/50 border-zinc-800/60">
          <CardContent className="p-4 text-center">
            <Bell className="w-8 h-8 mx-auto mb-2 text-zinc-400" />
            <p className="text-3xl font-mono font-bold text-zinc-100">{alerts.length}</p>
            <p className="text-xs text-zinc-500 uppercase tracking-wider">Total Alerts</p>
          </CardContent>
        </Card>
        <Card className="bg-red-950/20 border-red-900/30">
          <CardContent className="p-4 text-center">
            <XCircle className="w-8 h-8 mx-auto mb-2 text-red-400" />
            <p className="text-3xl font-mono font-bold text-red-400">{criticalAlerts.length}</p>
            <p className="text-xs text-zinc-500 uppercase tracking-wider">Critical</p>
          </CardContent>
        </Card>
        <Card className="bg-yellow-950/20 border-yellow-900/30">
          <CardContent className="p-4 text-center">
            <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
            <p className="text-3xl font-mono font-bold text-yellow-400">{warningAlerts.length}</p>
            <p className="text-xs text-zinc-500 uppercase tracking-wider">Warnings</p>
          </CardContent>
        </Card>
      </div>

      {/* Alert List */}
      {alerts.length > 0 ? (
        <ScrollArea className="h-[500px]">
          <div className="space-y-3 pr-4">
            {alerts.map((alert) => (
              <AlertCard 
                key={alert.id} 
                alert={alert} 
                onAcknowledge={onAcknowledge}
              />
            ))}
          </div>
        </ScrollArea>
      ) : (
        <Card className="bg-zinc-950/50 border-zinc-800/60 p-12 text-center">
          <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">All Clear!</h3>
          <p className="text-zinc-500">No unacknowledged alerts at this time</p>
        </Card>
      )}
    </div>
  );
};
