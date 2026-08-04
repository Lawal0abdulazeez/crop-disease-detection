/**
 * API base URL
 * - Local dev: Vite proxy uses /api → http://127.0.0.1:8000
 * - Production: set VITE_API_URL to your Render backend URL
 */
const API_BASE =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "/api";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse(res);
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  return handleResponse(res);
}

export async function getClasses() {
  const res = await fetch(`${API_BASE}/classes`);
  return handleResponse(res);
}

export async function predictImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: form,
  });
  return handleResponse(res);
}

export { API_BASE };
