export type Role = "admin" | "operator" | "reviewer" | "viewer";

export function canReview(role: string | undefined): boolean {
  return role === "admin" || role === "reviewer";
}

export function canManageUsers(role: string | undefined): boolean {
  return role === "admin";
}

export function canManageCamerasWrite(role: string | undefined): boolean {
  return role === "admin";
}

/** Старт/стоп live-анализа (admin + operator). */
export function canControlLiveAnalysis(role: string | undefined): boolean {
  return role === "admin" || role === "operator";
}

export function canAddOperatorNotes(role: string | undefined): boolean {
  return role === "admin" || role === "operator";
}
