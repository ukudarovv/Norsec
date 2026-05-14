import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { createUser, listUsers, type UserRow } from "../api/client";
import { useAuth } from "../auth/AuthProvider";
import { canManageUsers } from "../auth/permissions";

export default function UsersPage() {
  const { user } = useAuth();
  const [rows, setRows] = useState<UserRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("viewer");

  const load = useCallback(async () => {
    setErr(null);
    try {
      setRows(await listUsers());
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed");
    }
  }, []);

  useEffect(() => {
    if (canManageUsers(user?.role)) void load();
  }, [load, user?.role]);

  if (!canManageUsers(user?.role)) {
    return <p className="text-slate-500">Admin only.</p>;
  }

  return (
    <div className="space-y-6">
      <Link className="text-sm text-sky-400 hover:underline" to="/">
        ← Dashboard
      </Link>
      <h1 className="text-xl font-semibold text-white">Users</h1>
      {err && <p className="text-sm text-red-400">{err}</p>}
      <form
        className="grid gap-2 rounded border border-slate-800 bg-slate-900/40 p-4 sm:grid-cols-2"
        onSubmit={(e) => {
          e.preventDefault();
          void (async () => {
            try {
              await createUser({ email, password, role });
              setEmail("");
              setPassword("");
              await load();
            } catch (ex) {
              setErr(ex instanceof Error ? ex.message : "Create failed");
            }
          })();
        }}
      >
        <input className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm" placeholder="email" value={email} onChange={(e) => setEmail(e.target.value)} />
        <input type="password" className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm" placeholder="password" value={password} onChange={(e) => setPassword(e.target.value)} />
        <select className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm" value={role} onChange={(e) => setRole(e.target.value)}>
          <option value="viewer">viewer</option>
          <option value="operator">operator</option>
          <option value="reviewer">reviewer</option>
          <option value="admin">admin</option>
        </select>
        <button type="submit" className="rounded bg-sky-700 px-3 py-1 text-sm text-white">
          Create user
        </button>
      </form>
      <ul className="space-y-2 text-sm">
        {rows.map((u) => (
          <li key={u.id} className="rounded border border-slate-800 px-3 py-2 font-mono text-xs text-slate-300">
            {u.email} — {u.role} — {u.is_active ? "active" : "disabled"}
          </li>
        ))}
      </ul>
    </div>
  );
}
