import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth/AuthProvider";
import ProtectedRoute from "./auth/ProtectedRoute";
import { canManageUsers } from "./auth/permissions";
import AnalyticsPage from "./pages/AnalyticsPage";
import CameraStatusPage from "./pages/CameraStatusPage";
import CamerasPage from "./pages/CamerasPage";
import DashboardPage from "./pages/DashboardPage";
import IncidentDetailPage from "./pages/IncidentDetailPage";
import IncidentsPage from "./pages/IncidentsPage";
import LiveMonitoringPage from "./pages/LiveMonitoringPage";
import LoginPage from "./pages/LoginPage";
import ReviewQueuePage from "./pages/ReviewQueuePage";
import UsersPage from "./pages/UsersPage";

function Shell() {
  const { user, logout, loading } = useAuth();
  const nav = useNavigate();

  if (loading) {
    return <p className="p-6 text-slate-400">Loading…</p>;
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div>
            <Link to="/" className="text-lg font-semibold text-white hover:text-sky-300">
              Incident review
            </Link>
            <p className="text-xs text-slate-400">
              AI risk candidates — requires human review. Not a bullying verdict.
            </p>
          </div>
          <nav className="flex flex-wrap items-center gap-4 text-sm">
            {user ? (
              <>
                <Link className="text-slate-300 hover:text-white" to="/">
                  Dashboard
                </Link>
                <Link className="text-slate-300 hover:text-white" to="/incidents">
                  Incidents
                </Link>
                <Link className="text-slate-300 hover:text-white" to="/review-queue">
                  Review queue
                </Link>
                <Link className="text-slate-300 hover:text-white" to="/analytics">
                  Analytics
                </Link>
                <Link className="text-slate-300 hover:text-white" to="/cameras-ui">
                  Cameras
                </Link>
                <Link className="text-slate-300 hover:text-white" to="/live">
                  Live
                </Link>
                {canManageUsers(user.role) && (
                  <Link className="text-slate-300 hover:text-white" to="/users">
                    Users
                  </Link>
                )}
                <span className="text-xs text-slate-500">
                  {user.email} ({user.role})
                </span>
                <button
                  type="button"
                  className="text-sky-400 hover:underline"
                  onClick={() => {
                    logout();
                    nav("/login");
                  }}
                >
                  Logout
                </button>
              </>
            ) : (
              <Link className="text-sky-400 hover:underline" to="/login">
                Login
              </Link>
            )}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/incidents"
            element={
              <ProtectedRoute>
                <IncidentsPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/incidents/:id"
            element={
              <ProtectedRoute>
                <IncidentDetailPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/review-queue"
            element={
              <ProtectedRoute>
                <ReviewQueuePage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/analytics"
            element={
              <ProtectedRoute>
                <AnalyticsPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cameras-ui"
            element={
              <ProtectedRoute>
                <CamerasPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cameras"
            element={
              <ProtectedRoute>
                <CameraStatusPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cameras/:cameraId"
            element={
              <ProtectedRoute>
                <CameraStatusPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/live"
            element={
              <ProtectedRoute>
                <LiveMonitoringPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/users"
            element={
              <ProtectedRoute>
                <UsersPage />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  );
}
