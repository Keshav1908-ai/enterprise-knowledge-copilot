// Thin client for the FastAPI backend. Every screen reads from these calls — no mock data.
const BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, { status, code, details } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request(path, { method = "GET", body, form, signal } = {}) {
  const init = { method, signal, headers: {} };
  if (form) init.body = form;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers["Content-Type"] = "application/json";
  }
  let res;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("Can't reach the API server. Check that the backend is running.", { status: 0 });
  }
  const text = await res.text();
  const data = text ? safeJson(text) : null;
  if (!res.ok) {
    const e = data?.error;
    throw new ApiError(e?.message || `Request failed (${res.status})`, {
      status: res.status,
      code: e?.code,
      details: e?.details,
    });
  }
  return data;
}

function safeJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

export const api = {
  health: () => request("/health"),
  config: () => request("/api/v1/config"),
  dashboard: () => request("/api/v1/dashboard"),

  listDocuments: () => request("/api/v1/documents"),
  getDocument: (id) => request(`/api/v1/documents/${id}`),
  deleteDocument: (id) => request(`/api/v1/documents/${id}`, { method: "DELETE" }),
  reindexDocument: (id) => request(`/api/v1/documents/${id}/reindex`, { method: "POST" }),
  uploadDocuments: (files) => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    return request("/api/v1/documents/upload", { method: "POST", form });
  },

  chat: (payload, signal) => request("/api/v1/chat", { method: "POST", body: payload, signal }),
  listConversations: () => request("/api/v1/conversations"),
  getConversation: (id) => request(`/api/v1/conversations/${id}`),
  deleteConversation: (id) => request(`/api/v1/conversations/${id}`, { method: "DELETE" }),

  metrics: ({ hours = 168, source = "chat", bucket = "hour" } = {}) =>
    request(`/api/v1/metrics?hours=${hours}&source=${source}&bucket=${bucket}`),

  runEvaluation: (payload) => request("/api/v1/evaluation/run", { method: "POST", body: payload }),
  evaluationResults: () => request("/api/v1/evaluation/results"),
  evaluationRun: (id) => request(`/api/v1/evaluation/results/${id}`),
};
