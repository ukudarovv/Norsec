const base = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "";

const TOKEN_KEY = "bullying_ai_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function api(path: string, init: RequestInit = {}) {
  const headers: Record<string, string> = {
    ...authHeaders(),
    ...(init.headers as Record<string, string> | undefined),
  };
  if (init.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const r = await fetch(`${base}${path}`, { ...init, headers });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  if (r.status === 204) return null;
  const ct = r.headers.get("content-type");
  if (ct && ct.includes("application/json")) {
    return r.json();
  }
  return null;
}

export type MeUser = { id: string; email: string; role: string; full_name: string | null };

export async function loginRequest(email: string, password: string): Promise<string> {
  const data = (await api("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
    headers: { "Content-Type": "application/json" },
  })) as { access_token: string };
  return data.access_token;
}

export async function fetchMe(): Promise<MeUser | null> {
  if (!getToken()) return null;
  try {
    return (await api("/api/auth/me")) as MeUser;
  } catch {
    return null;
  }
}

export type Incident = {
  id: string;
  camera_id: string;
  camera_external_key?: string | null;
  start_sec: number;
  end_sec: number;
  risk_score: number;
  risk_level: string;
  signal_types: string[];
  explanation: string[];
  review_status: string;
  evidence: Record<string, unknown>;
  involved_track_ids?: number[] | null;
  clip_path?: string | null;
  created_at: string;
  last_reviewer_email?: string | null;
};

export async function fetchIncidents(params: Record<string, string | undefined>) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v) q.set(k, v);
  });
  const path = `/api/incidents${q.toString() ? `?${q}` : ""}`;
  return (await api(path)) as Incident[];
}

export async function fetchIncident(id: string) {
  const r = await fetch(`${base}/api/incidents/${encodeURIComponent(id)}`, { headers: authHeaders() });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as Incident;
}

export type ReviewRow = {
  id: string;
  incident_id: string;
  reviewer_id: string;
  status: string;
  comment?: string | null;
  tags?: string[];
  created_at: string;
};

export type IncidentDetailPayload = {
  incident: Incident;
  evidence: Record<string, unknown>;
  reviews: ReviewRow[];
  video_clip_url: string;
  snapshot_url: string;
  analytics: {
    social_signals: unknown[];
    pose_signals: unknown[];
    action_signals: unknown[];
    audio_signals: unknown[];
    suppression_reasons: string[];
  };
};

export async function fetchIncidentDetail(id: string): Promise<IncidentDetailPayload | null> {
  const r = await fetch(`${base}/api/incidents/${encodeURIComponent(id)}/detail`, { headers: authHeaders() });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as IncidentDetailPayload;
}

export type DashboardStats = {
  totals: {
    risk_candidates: number;
    needs_review: number;
    confirmed: number;
    false_positives: number;
    average_risk_score: number;
    false_positive_rate: number;
  };
  by_status: Record<string, number>;
  by_risk_level: Record<string, number>;
  by_day: Record<string, number>;
  by_camera: Record<string, number>;
  camera_health: { total_cameras: number; online_or_active: number };
};

export async function fetchDashboardStats(): Promise<DashboardStats> {
  return (await api("/api/dashboard/stats")) as DashboardStats;
}

export async function fetchReviewQueue(): Promise<Incident[]> {
  return (await api("/api/reviews/queue")) as Incident[];
}

export async function postOperatorNote(id: string, comment: string): Promise<Incident> {
  return (await api(`/api/incidents/${encodeURIComponent(id)}/notes`, {
    method: "POST",
    body: JSON.stringify({ comment }),
  })) as Incident;
}

export function incidentMediaUrl(incidentId: string, kind: "clip" | "snapshot"): string {
  const b = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "";
  const t = getToken();
  const q = t ? `?token=${encodeURIComponent(t)}` : "";
  const path = kind === "clip" ? "clip" : "snapshot";
  return `${b}/api/incidents/${encodeURIComponent(incidentId)}/media/${path}${q}`;
}

export type IncidentAnalytics = {
  incident_id: string;
  camera_id: string;
  social_signals: unknown[];
  pose_signals: unknown[];
  trajectory_summary: Record<string, unknown>;
  suppression: unknown;
  evidence: Record<string, unknown>;
  note?: string;
};

export async function fetchIncidentAnalytics(id: string): Promise<IncidentAnalytics | null> {
  const r = await fetch(`${base}/api/incidents/${encodeURIComponent(id)}/analytics`, { headers: authHeaders() });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as IncidentAnalytics;
}

export type LiveCameraAnalytics = {
  camera_id: string;
  overlay_seq: number;
  risk: { score?: number; level?: string };
  active_social_signals: unknown[];
  active_pose_signals: unknown[];
  trajectory_preview: unknown[];
  risk_modifiers: unknown[];
  suppression: Record<string, unknown>;
  signal_catalog: { social: string[]; pose: string[] };
};

export async function fetchCameraAnalyticsLive(cameraId: string): Promise<LiveCameraAnalytics | null> {
  const r = await fetch(`${base}/api/cameras/${encodeURIComponent(cameraId)}/analytics/live`, {
    headers: authHeaders(),
  });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as LiveCameraAnalytics;
}

export async function fetchAnalyticsSignals(): Promise<{
  social_signals: string[];
  pose_signals: string[];
  suppression_rules: string[];
}> {
  return (await api("/api/analytics/signals")) as {
    social_signals: string[];
    pose_signals: string[];
    suppression_rules: string[];
  };
}

export async function submitReview(id: string, body: { status: string; comment?: string; tags?: string[] }) {
  return (await api(`/api/incidents/${encodeURIComponent(id)}/review`, {
    method: "POST",
    body: JSON.stringify(body),
  })) as Incident;
}

export type CameraRow = {
  id: string;
  name: string;
  location: string | null;
  rtsp_url: string | null;
  status: string;
  is_active: boolean;
  external_key: string | null;
  created_at: string;
};

export async function listCamerasDetailed(): Promise<CameraRow[]> {
  return (await api("/api/cameras")) as CameraRow[];
}

export async function createCamera(body: { name: string; external_key?: string | null; rtsp_url?: string | null }) {
  return (await api("/api/cameras", { method: "POST", body: JSON.stringify(body) })) as CameraRow;
}

export async function fetchCameraStatus(cameraId: string) {
  return (await api(`/api/cameras/${encodeURIComponent(cameraId)}/status`)) as Record<string, unknown>;
}

export async function listRunningCameras(): Promise<string[]> {
  const r = (await api("/api/cameras/running")) as { cameras?: string[] };
  return r.cameras ?? [];
}

export async function testCameraConnection(cameraId: string) {
  return (await api(`/api/cameras/${encodeURIComponent(cameraId)}/test-connection`, {
    method: "POST",
  })) as { ok: boolean; detail: string };
}

export async function startCameraAnalysis(cameraId: string) {
  return (await api(`/api/cameras/${encodeURIComponent(cameraId)}/start`, { method: "POST" })) as {
    running: boolean;
    camera_id: string;
  };
}

export async function stopCameraAnalysis(cameraId: string) {
  return (await api(`/api/cameras/${encodeURIComponent(cameraId)}/stop`, { method: "POST" })) as {
    stopped: boolean;
    camera_id: string;
  };
}

export async function restartCameraAnalysis(cameraId: string) {
  return (await api(`/api/cameras/${encodeURIComponent(cameraId)}/restart`, { method: "POST" })) as {
    running: boolean;
    camera_id: string;
  };
}

export function mjpegUrl(cameraId: string): string {
  const base = import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "";
  const t = getToken();
  const q = t ? `?token=${encodeURIComponent(t)}` : "";
  return `${base}/api/cameras/${encodeURIComponent(cameraId)}/mjpeg${q}`;
}

export function overlayWsUrl(cameraId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  const t = getToken();
  const q = t ? `?token=${encodeURIComponent(t)}` : "";
  return `${proto}//${host}/ws/cameras/${encodeURIComponent(cameraId)}/overlay${q}`;
}

export type OverlayPayload = {
  camera_id: string;
  timestamp: string;
  people: Array<{ track_id: number; bbox: number[]; confidence?: number; skeleton?: number[][] }>;
  poses: unknown[];
  signals: Array<{ type?: string; name?: string; severity?: number }>;
  risk: { score: number; level: string };
  analytics?: {
    social?: Array<Record<string, unknown>>;
    pose?: Array<Record<string, unknown>>;
    trajectory_preview?: Array<{ track_id: number; points: number[][] }>;
    suppression?: Record<string, unknown>;
    risk_modifiers?: string[];
  };
};

export type UserRow = { id: string; email: string; role: string; full_name: string | null; is_active: boolean };

export async function listUsers(): Promise<UserRow[]> {
  return (await api("/api/users")) as UserRow[];
}

export async function createUser(body: { email: string; password: string; role: string; full_name?: string | null }) {
  return (await api("/api/users", { method: "POST", body: JSON.stringify(body) })) as UserRow;
}
