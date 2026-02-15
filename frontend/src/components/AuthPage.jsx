import { useState } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Activity,
  Mail,
  Lock,
  User,
  Eye,
  EyeOff,
  ArrowRight,
  Shield
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";

export const AuthPage = ({ onLogin, API }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: "",
    confirmPassword: ""
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        const response = await axios.post(`${API}/auth/login`, {
          email: formData.email,
          password: formData.password
        });
        
        const { access_token, user } = response.data;
        localStorage.setItem("token", access_token);
        localStorage.setItem("user", JSON.stringify(user));
        toast.success(`Welcome back, ${user.name}!`);
        onLogin(user, access_token);
      } else {
        if (formData.password !== formData.confirmPassword) {
          toast.error("Passwords don't match");
          setLoading(false);
          return;
        }
        
        if (formData.password.length < 6) {
          toast.error("Password must be at least 6 characters");
          setLoading(false);
          return;
        }

        const response = await axios.post(`${API}/auth/register`, {
          email: formData.email,
          password: formData.password,
          name: formData.name
        });
        
        const { access_token, user } = response.data;
        localStorage.setItem("token", access_token);
        localStorage.setItem("user", JSON.stringify(user));
        toast.success(`Welcome, ${user.name}! Account created successfully.`);
        onLogin(user, access_token);
      }
    } catch (error) {
      const message = error.response?.data?.detail || "Authentication failed";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center p-4" data-testid="auth-page">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-cyan-900/20 via-zinc-950 to-zinc-950" />
      </div>

      <div className="relative w-full max-w-md">
        <motion.div 
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-xl bg-cyan-500/10 border border-cyan-500/20 mb-4">
            <Activity className="w-8 h-8 text-cyan-400" />
          </div>
          <h1 className="text-3xl font-bold text-zinc-100 tracking-tight">PredictMaint</h1>
          <p className="text-zinc-500 mt-2">Multimodal Predictive Maintenance System</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="bg-zinc-900/80 border-zinc-800/60 backdrop-blur-sm">
            <CardHeader className="space-y-1 pb-4">
              <CardTitle className="text-xl text-zinc-100">
                {isLogin ? "Sign in to your account" : "Create an account"}
              </CardTitle>
              <CardDescription className="text-zinc-500">
                {isLogin 
                  ? "Enter your credentials to access the dashboard" 
                  : "Enter your details to create your account"}
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                {!isLogin && (
                  <div className="space-y-2">
                    <Label htmlFor="name" className="text-zinc-300">Full Name</Label>
                    <div className="relative">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                      <Input
                        id="name"
                        name="name"
                        type="text"
                        placeholder="John Doe"
                        value={formData.name}
                        onChange={handleChange}
                        required={!isLogin}
                        className="pl-10 bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-600"
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
                      name="email"
                      type="email"
                      placeholder="you@example.com"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="pl-10 bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-600"
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
                      name="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      value={formData.password}
                      onChange={handleChange}
                      required
                      className="pl-10 pr-10 bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-600"
                      data-testid="password-input"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                {!isLogin && (
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword" className="text-zinc-300">Confirm Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                      <Input
                        id="confirmPassword"
                        name="confirmPassword"
                        type={showPassword ? "text" : "password"}
                        placeholder="••••••••"
                        value={formData.confirmPassword}
                        onChange={handleChange}
                        required={!isLogin}
                        className="pl-10 bg-zinc-800/50 border-zinc-700 text-zinc-100 placeholder:text-zinc-600"
                        data-testid="confirm-password-input"
                      />
                    </div>
                  </div>
                )}

                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-cyan-500 text-black hover:bg-cyan-400 font-medium"
                  data-testid="auth-submit-btn"
                >
                  {loading ? (
                    <span className="flex items-center gap-2">
                      <span className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                      {isLogin ? "Signing in..." : "Creating account..."}
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      {isLogin ? "Sign In" : "Create Account"}
                      <ArrowRight className="w-4 h-4" />
                    </span>
                  )}
                </Button>
              </form>

              <div className="mt-6 text-center">
                <button
                  type="button"
                  onClick={() => {
                    setIsLogin(!isLogin);
                    setFormData({ email: "", password: "", name: "", confirmPassword: "" });
                  }}
                  className="text-sm text-zinc-500 hover:text-cyan-400 transition-colors"
                  data-testid="toggle-auth-mode"
                >
                  {isLogin 
                    ? "Don't have an account? Sign up" 
                    : "Already have an account? Sign in"}
                </button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div 
          className="mt-8 grid grid-cols-3 gap-4 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div className="p-3">
            <Shield className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
            <p className="text-xs text-zinc-500">Secure Auth</p>
          </div>
          <div className="p-3">
            <Activity className="w-5 h-5 text-emerald-400 mx-auto mb-2" />
            <p className="text-xs text-zinc-500">Real-time Data</p>
          </div>
          <div className="p-3">
            <User className="w-5 h-5 text-violet-400 mx-auto mb-2" />
            <p className="text-xs text-zinc-500">Multi-tenant</p>
          </div>
        </motion.div>
      </div>
    </div>
  );
};
