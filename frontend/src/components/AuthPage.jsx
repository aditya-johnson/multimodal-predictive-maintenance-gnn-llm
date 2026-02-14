import { useState } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import {
  Activity,
  Mail,
  Lock,
  User,
  LogIn,
  UserPlus,
  ArrowRight
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

export const AuthPage = ({ onLogin, API }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: ""
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const endpoint = isLogin ? "/auth/login" : "/auth/register";
      const payload = isLogin 
        ? { email: formData.email, password: formData.password }
        : formData;

      const response = await axios.post(`${API}${endpoint}`, payload);
      
      const { access_token, user } = response.data;
      
      // Store token
      localStorage.setItem("token", access_token);
      localStorage.setItem("user", JSON.stringify(user));
      
      toast.success(isLogin ? "Welcome back!" : "Account created successfully!");
      onLogin(user, access_token);
      
    } catch (error) {
      const message = error.response?.data?.detail || "Authentication failed";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-4" data-testid="auth-page">
      <div className="w-full max-w-md">
        {/* Logo */}
        <motion.div 
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="w-16 h-16 mx-auto rounded-xl bg-cyan-500/10 flex items-center justify-center mb-4">
            <Activity className="w-8 h-8 text-cyan-400" />
          </div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">PredictMaint</h1>
          <p className="text-zinc-500 mt-2">Multimodal Predictive Maintenance</p>
        </motion.div>

        {/* Auth Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader className="text-center">
              <CardTitle className="text-xl text-zinc-100">
                {isLogin ? "Sign In" : "Create Account"}
              </CardTitle>
              <CardDescription className="text-zinc-500">
                {isLogin 
                  ? "Enter your credentials to access your dashboard" 
                  : "Register to start monitoring your equipment"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                {!isLogin && (
                  <div className="space-y-2">
                    <Label htmlFor="name" className="text-zinc-300">Name</Label>
                    <div className="relative">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                      <Input
                        id="name"
                        type="text"
                        placeholder="John Doe"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        className="pl-10 bg-zinc-800/50 border-zinc-700 text-zinc-100"
                        required={!isLogin}
                        data-testid="name-input"
                      />
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="email" className="text-zinc-300">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      className="pl-10 bg-zinc-800/50 border-zinc-700 text-zinc-100"
                      required
                      data-testid="email-input"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password" className="text-zinc-300">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                    <Input
                      id="password"
                      type="password"
                      placeholder="••••••••"
                      value={formData.password}
                      onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                      className="pl-10 bg-zinc-800/50 border-zinc-700 text-zinc-100"
                      required
                      minLength={6}
                      data-testid="password-input"
                    />
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-cyan-500 text-black hover:bg-cyan-400 font-medium"
                  disabled={loading}
                  data-testid="auth-submit-btn"
                >
                  {loading ? (
                    <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-black" />
                  ) : (
                    <>
                      {isLogin ? <LogIn className="w-4 h-4 mr-2" /> : <UserPlus className="w-4 h-4 mr-2" />}
                      {isLogin ? "Sign In" : "Create Account"}
                    </>
                  )}
                </Button>
              </form>

              <div className="mt-6 text-center">
                <button
                  type="button"
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-sm text-zinc-400 hover:text-cyan-400 transition-colors"
                  data-testid="toggle-auth-mode"
                >
                  {isLogin ? "Don't have an account? " : "Already have an account? "}
                  <span className="font-medium">{isLogin ? "Sign up" : "Sign in"}</span>
                </button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Features */}
        <motion.div 
          className="mt-8 grid grid-cols-3 gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <div className="text-center">
            <div className="text-cyan-400 text-2xl font-bold">GNN</div>
            <p className="text-xs text-zinc-500">Graph Neural Net</p>
          </div>
          <div className="text-center">
            <div className="text-cyan-400 text-2xl font-bold">NLP</div>
            <p className="text-xs text-zinc-500">Log Analysis</p>
          </div>
          <div className="text-center">
            <div className="text-cyan-400 text-2xl font-bold">98%</div>
            <p className="text-xs text-zinc-500">Accuracy</p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};
