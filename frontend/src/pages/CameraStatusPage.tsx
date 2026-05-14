import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { fetchCameraStatus, listCamerasDetailed, type CameraRow } from "../api/client";

export default function CameraStatusPage() {
  const { cameraId } = useParams<{ cameraId?: string }>();
  const [cams, setCams] = useState<CameraRow[]>([]);
  const [status, setStatus] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      setErr(null);
      try {
        const list = await listCamerasDetailed();
        setCams(list);
        if (cameraId) {
          setStatus(await fetchCameraStatus(cameraId));
        } else {
          setStatus(null);
        }
      } catch (e) {
        setErr(e instanceof Error ? e.message : "Failed");
      }
    })();
  }, [cameraId]);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-white">Cameras</h1>
      <p className="text-sm text-slate-400">Registered cameras (RTSP hidden for viewer role in list).</p>
      {err && <p className="text-sm text-red-400">{err}</p>}
      <ul className="space-y-2">
        {cams.map((c) => (
          <li key={c.id}>
            <Link className="text-sky-400 hover:underline" to={`/cameras/${encodeURIComponent(c.id)}`}>
              {c.name}
            </Link>
            <span className="ml-2 text-xs text-slate-500">{c.external_key || ""}</span>
          </li>
        ))}
      </ul>
      {cams.length === 0 && <p className="text-slate-500">No cameras yet.</p>}
      {cameraId && status && (
        <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="mb-2 text-sm font-semibold text-slate-200">Status</h2>
          <pre className="text-xs text-slate-300">{JSON.stringify(status, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
