import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Building2,
  Users,
  Plus,
  Mail,
  Shield,
  UserCog,
  Eye,
  Crown,
  Check,
  X,
  ChevronRight,
  Settings
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
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

const ROLE_CONFIG = {
  admin: { icon: Crown, color: "text-yellow-400 bg-yellow-950/30 border-yellow-900/50", label: "Admin" },
  operator: { icon: UserCog, color: "text-cyan-400 bg-cyan-950/30 border-cyan-900/50", label: "Operator" },
  viewer: { icon: Eye, color: "text-zinc-400 bg-zinc-800/30 border-zinc-700/50", label: "Viewer" }
};

const RoleBadge = ({ role }) => {
  const config = ROLE_CONFIG[role] || ROLE_CONFIG.viewer;
  const Icon = config.icon;
  return (
    <Badge className={`${config.color} border text-xs`}>
      <Icon className="w-3 h-3 mr-1" />
      {config.label}
    </Badge>
  );
};

export const OrganizationManager = ({ currentOrg, onOrgChange, API }) => {
  const [organizations, setOrganizations] = useState([]);
  const [members, setMembers] = useState([]);
  const [invitations, setInvitations] = useState([]);
  const [pendingInvites, setPendingInvites] = useState([]);
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [newOrg, setNewOrg] = useState({ name: "", description: "" });
  const [newInvite, setNewInvite] = useState({ email: "", role: "operator" });

  useEffect(() => {
    fetchOrganizations();
    fetchPendingInvitations();
  }, []);

  useEffect(() => {
    if (currentOrg) {
      fetchMembers();
    }
  }, [currentOrg]);

  const fetchOrganizations = async () => {
    try {
      const response = await axios.get(`${API}/organizations`);
      setOrganizations(response.data);
    } catch (error) {
      console.error("Error fetching orgs:", error);
    }
  };

  const fetchMembers = async () => {
    if (!currentOrg) return;
    try {
      const response = await axios.get(`${API}/organizations/${currentOrg.id}/members`);
      setMembers(response.data);
    } catch (error) {
      console.error("Error fetching members:", error);
    }
  };

  const fetchPendingInvitations = async () => {
    try {
      const response = await axios.get(`${API}/invitations`);
      setPendingInvites(response.data);
    } catch (error) {
      console.error("Error fetching invitations:", error);
    }
  };

  const createOrganization = async () => {
    if (!newOrg.name) {
      toast.error("Organization name is required");
      return;
    }
    setLoading(true);
    try {
      const response = await axios.post(`${API}/organizations`, newOrg);
      toast.success("Organization created!");
      setCreateDialogOpen(false);
      setNewOrg({ name: "", description: "" });
      await fetchOrganizations();
      onOrgChange(response.data.organization, response.data.role);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to create organization");
    } finally {
      setLoading(false);
    }
  };

  const switchOrganization = async (orgId) => {
    try {
      const response = await axios.post(`${API}/organizations/${orgId}/switch`);
      localStorage.setItem("token", response.data.access_token);
      toast.success(`Switched to ${response.data.organization.name}`);
      onOrgChange(response.data.organization, response.data.role);
    } catch (error) {
      toast.error("Failed to switch organization");
    }
  };

  const inviteMember = async () => {
    if (!newInvite.email) {
      toast.error("Email is required");
      return;
    }
    setLoading(true);
    try {
      await axios.post(`${API}/organizations/${currentOrg.id}/invite`, newInvite);
      toast.success(`Invitation sent to ${newInvite.email}`);
      setInviteDialogOpen(false);
      setNewInvite({ email: "", role: "operator" });
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to send invitation");
    } finally {
      setLoading(false);
    }
  };

  const acceptInvitation = async (inviteId) => {
    try {
      const response = await axios.post(`${API}/invitations/${inviteId}/accept`);
      toast.success("Invitation accepted!");
      await fetchOrganizations();
      await fetchPendingInvitations();
      switchOrganization(response.data.org_id);
    } catch (error) {
      toast.error("Failed to accept invitation");
    }
  };

  const updateMemberRole = async (userId, newRole) => {
    try {
      await axios.put(`${API}/organizations/${currentOrg.id}/members/${userId}/role?role=${newRole}`);
      toast.success("Role updated");
      fetchMembers();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to update role");
    }
  };

  const removeMember = async (userId) => {
    try {
      await axios.delete(`${API}/organizations/${currentOrg.id}/members/${userId}`);
      toast.success("Member removed");
      fetchMembers();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to remove member");
    }
  };

  return (
    <div className="space-y-6" data-testid="org-manager">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">Organization</h1>
          <p className="text-zinc-500 mt-1">Manage your team and access control</p>
        </div>
        
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button className="bg-cyan-500 text-black hover:bg-cyan-400" data-testid="create-org-btn">
              <Plus className="w-4 h-4 mr-2" />
              New Organization
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-zinc-900 border-zinc-800">
            <DialogHeader>
              <DialogTitle className="text-zinc-100">Create Organization</DialogTitle>
              <DialogDescription className="text-zinc-500">
                Create a new organization to manage machines and team members
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label className="text-zinc-300">Organization Name</Label>
                <Input
                  placeholder="Acme Manufacturing"
                  value={newOrg.name}
                  onChange={(e) => setNewOrg({ ...newOrg, name: e.target.value })}
                  className="bg-zinc-800/50 border-zinc-700 text-zinc-100"
                  data-testid="org-name-input"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-zinc-300">Description (optional)</Label>
                <Input
                  placeholder="Main production facility"
                  value={newOrg.description}
                  onChange={(e) => setNewOrg({ ...newOrg, description: e.target.value })}
                  className="bg-zinc-800/50 border-zinc-700 text-zinc-100"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCreateDialogOpen(false)} className="border-zinc-700">
                Cancel
              </Button>
              <Button onClick={createOrganization} disabled={loading} className="bg-cyan-500 text-black hover:bg-cyan-400">
                {loading ? "Creating..." : "Create Organization"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Pending Invitations */}
      {pendingInvites.length > 0 && (
        <Card className="bg-cyan-950/20 border-cyan-900/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-cyan-400 flex items-center gap-2 text-sm">
              <Mail className="w-4 h-4" />
              Pending Invitations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {pendingInvites.map((invite) => (
              <div key={invite.id} className="flex items-center justify-between p-3 bg-zinc-900/50 rounded-md">
                <div>
                  <p className="text-zinc-100 font-medium">{invite.org_name}</p>
                  <p className="text-xs text-zinc-500">Role: {invite.role}</p>
                </div>
                <Button size="sm" onClick={() => acceptInvitation(invite.id)} className="bg-cyan-500 text-black hover:bg-cyan-400">
                  <Check className="w-4 h-4 mr-1" />
                  Accept
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Current Organization */}
      {currentOrg && (
        <Card className="bg-zinc-950/50 border-cyan-500/30">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-lg bg-cyan-500/10 flex items-center justify-center">
                  <Building2 className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <CardTitle className="text-zinc-100">{currentOrg.name}</CardTitle>
                  <CardDescription className="text-zinc-500">{currentOrg.description || "No description"}</CardDescription>
                </div>
              </div>
              <RoleBadge role={currentOrg.role} />
            </div>
          </CardHeader>
          <CardContent>
            {/* Members Section */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-zinc-100 flex items-center gap-2">
                <Users className="w-5 h-5 text-zinc-400" />
                Team Members ({members.length})
              </h3>
              
              {currentOrg.role === "admin" && (
                <Dialog open={inviteDialogOpen} onOpenChange={setInviteDialogOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="border-zinc-700" data-testid="invite-btn">
                      <Mail className="w-4 h-4 mr-2" />
                      Invite Member
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="bg-zinc-900 border-zinc-800">
                    <DialogHeader>
                      <DialogTitle className="text-zinc-100">Invite Team Member</DialogTitle>
                      <DialogDescription className="text-zinc-500">
                        Send an invitation to join your organization
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <div className="space-y-2">
                        <Label className="text-zinc-300">Email Address</Label>
                        <Input
                          type="email"
                          placeholder="colleague@company.com"
                          value={newInvite.email}
                          onChange={(e) => setNewInvite({ ...newInvite, email: e.target.value })}
                          className="bg-zinc-800/50 border-zinc-700 text-zinc-100"
                          data-testid="invite-email-input"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-zinc-300">Role</Label>
                        <Select value={newInvite.role} onValueChange={(v) => setNewInvite({ ...newInvite, role: v })}>
                          <SelectTrigger className="bg-zinc-800/50 border-zinc-700">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="bg-zinc-900 border-zinc-800">
                            <SelectItem value="admin">Admin - Full access</SelectItem>
                            <SelectItem value="operator">Operator - Manage machines & predictions</SelectItem>
                            <SelectItem value="viewer">Viewer - Read-only access</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setInviteDialogOpen(false)} className="border-zinc-700">
                        Cancel
                      </Button>
                      <Button onClick={inviteMember} disabled={loading} className="bg-cyan-500 text-black hover:bg-cyan-400">
                        Send Invitation
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              )}
            </div>

            <ScrollArea className="h-[300px]">
              <div className="space-y-2">
                {members.map((member) => (
                  <div key={member.id} className="flex items-center justify-between p-3 bg-zinc-900/50 rounded-md border border-zinc-800/60">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-zinc-800 flex items-center justify-center">
                        <span className="text-zinc-400 font-medium">{member.name?.[0]?.toUpperCase() || "?"}</span>
                      </div>
                      <div>
                        <p className="text-zinc-100 font-medium">{member.name}</p>
                        <p className="text-xs text-zinc-500">{member.email}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {currentOrg.role === "admin" ? (
                        <Select value={member.role} onValueChange={(v) => updateMemberRole(member.id, v)}>
                          <SelectTrigger className="w-[120px] bg-zinc-800/50 border-zinc-700 h-8">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent className="bg-zinc-900 border-zinc-800">
                            <SelectItem value="admin">Admin</SelectItem>
                            <SelectItem value="operator">Operator</SelectItem>
                            <SelectItem value="viewer">Viewer</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <RoleBadge role={member.role} />
                      )}
                      {currentOrg.role === "admin" && member.role !== "admin" && (
                        <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-500 hover:text-red-400"
                                onClick={() => removeMember(member.id)}>
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}

      {/* Organization List */}
      {organizations.length > 1 && (
        <Card className="bg-zinc-950/50 border-zinc-800/60">
          <CardHeader>
            <CardTitle className="text-zinc-100">Your Organizations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {organizations.map((org) => (
              <button
                key={org.id}
                className={`w-full p-3 rounded-md text-left transition-all flex items-center justify-between ${
                  currentOrg?.id === org.id 
                    ? "bg-cyan-500/10 border border-cyan-500/30" 
                    : "bg-zinc-900/50 border border-zinc-800/60 hover:border-zinc-700"
                }`}
                onClick={() => switchOrganization(org.id)}
              >
                <div className="flex items-center gap-3">
                  <Building2 className={`w-5 h-5 ${currentOrg?.id === org.id ? "text-cyan-400" : "text-zinc-500"}`} />
                  <div>
                    <p className="text-zinc-100 font-medium">{org.name}</p>
                    <p className="text-xs text-zinc-500">{org.description || "No description"}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <RoleBadge role={org.role} />
                  {currentOrg?.id === org.id && <Check className="w-4 h-4 text-cyan-400" />}
                </div>
              </button>
            ))}
          </CardContent>
        </Card>
      )}

      {/* RBAC Info */}
      <Card className="bg-zinc-900/30 border-zinc-800/40">
        <CardHeader>
          <CardTitle className="text-zinc-300 flex items-center gap-2 text-sm">
            <Shield className="w-4 h-4" />
            Role Permissions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Crown className="w-4 h-4 text-yellow-400" />
                <span className="text-zinc-100 font-medium">Admin</span>
              </div>
              <ul className="text-zinc-500 space-y-1 text-xs">
                <li>• Manage organization</li>
                <li>• Invite/remove users</li>
                <li>• All operator permissions</li>
              </ul>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <UserCog className="w-4 h-4 text-cyan-400" />
                <span className="text-zinc-100 font-medium">Operator</span>
              </div>
              <ul className="text-zinc-500 space-y-1 text-xs">
                <li>• Create/edit machines</li>
                <li>• Run predictions</li>
                <li>• Manage alerts</li>
              </ul>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Eye className="w-4 h-4 text-zinc-400" />
                <span className="text-zinc-100 font-medium">Viewer</span>
              </div>
              <ul className="text-zinc-500 space-y-1 text-xs">
                <li>• View dashboards</li>
                <li>• Download reports</li>
                <li>• Read-only access</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
