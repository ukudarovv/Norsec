import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { createCamera, listCamerasDetailed, type CameraRow } from "../api/client";
import { useAuth } from "../auth/AuthProvider";
import { canManageCamerasWrite } from "../auth/permissions";

export default function CamerasPage() {
  const { user } = useAuth();
  const [rows, setRows] = useState<CameraRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [externalKey, setExternalKey] = useState("");
  const [rtsp, setRtsp] = useState("");

  const load = useCallback(async () => {
    setErr(null);
    try {
      setRows(await listCamerasDetailed());
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-6">
      <Link className="text-sm text-sky-400 hover:underline" to="/">
        ← Dashboard
      </Link>
      <h1 className="text-xl font-semibold text-white">Cameras</h1>
      {err && <p className="text-sm text-red-400">{err}</p>}
      {canManageCamerasWrite(user?.role) && (
        <form
          className="grid gap-2 rounded border border-slate-800 bg-slate-900/40 p-4 sm:grid-cols-2"
          onSubmit={(e) => {
            e.preventDefault();
            void (async () => {
              try {
                await createCamera({ name, external_key: externalKey || null, rtsp_url: rtsp || null });
                setName("");
                setExternalKey("");
                setRtsp("");
                await load();
              } catch (ex) {
                setErr(ex instanceof Error ? ex.message : "Failed");
              }
            })();
          }}
        >
          <input className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm" placeholder="name" value={name} onChange={(e) => setName(e.target.value)} />
          <input className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm" placeholder="external_key (fusion id)" value={externalKey} onChange={(e) => setExternalKey(e.target.value)} />
          <input className="rounded border border-slate-700 bg-slate-950 px-2 py-1 text-sm sm:col-span-2" placeholder="rtsp_url (admin only in API)" value={rtsp} onChange={(e) => setRtsp(e.target.value)} />
          <button type="submit" className="rounded bg-sky-700 px-3 py-1 text-sm text-white sm:col-span-2">
            Add camera
          </button>
        </form>
      )}
      <ul className="space-y-2 text-sm">
        {rows.map((c) => (
          <li key={c.id} className="rounded border border-slate-800 px-3 py-2 text-slate-300">
            <Link className="text-sky-400 hover:underline" to={`/cameras/${c.id}`}>
              {c.name}
            </Link>{" "}
            <span className="text-xs text-slate-500">({c.external_key || "—"})</span>
            {c.rtsp_url && <span className="ml-2 text-xs text-slate-600">RTSP configured</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
