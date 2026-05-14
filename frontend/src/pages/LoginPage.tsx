import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthProvider";

export default function LoginPage() {
  const { login, user } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const from = (loc.state as { from?: string } | null)?.from || "/";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (user) {
      nav(from, { replace: true });
    }
  }, [user, from, nav]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    try {
      await login(email, password);
    } catch {
      setErr("Invalid email or password");
    }
  }

  return (
    <div className="mx-auto max-w-md rounded-lg border border-slate-800 bg-slate-900/50 p-6">
      <h1 className="mb-1 text-xl font-semibold text-white">Sign in</h1>
      <p className="mb-4 text-xs text-slate-500">AI risk candidates — human review required.</p>
      <form className="space-y-3" onSubmit={(e) => void onSubmit(e)}>
        <input
          type="email"
          required
          className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          required
          className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {err && <p className="text-sm text-red-400">{err}</p>}
        <button type="submit" className="w-full rounded bg-sky-700 py-2 text-sm font-medium text-white hover:bg-sky-600">
          Login
        </button>
      </form>
      <p className="mt-4 text-center text-xs text-slate-500">
        <Link className="text-sky-400 hover:underline" to="/">
          Back
        </Link>
      </p>
    </div>
  );
}
