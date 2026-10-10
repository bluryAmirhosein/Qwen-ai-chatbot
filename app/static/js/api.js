// api.js — thin async wrapper around the JSON chatbot backend.
// The template never renders backend data server-side; every value
// shown in the page comes from one of these calls, made client-side.


const API_PREFIX = window.API_PREFIX || "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${API_PREFIX}${path}`, options);

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }

  if (res.status === 204) return null;
  return res.json();
}

/**
 * Builds a query string. Array values are sent as repeated keys
 * (e.g. document_ids=3&document_ids=5), which is what FastAPI expects
 * for list query parameters. Empty/undefined values are skipped.
 */
function buildQuery(params) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      value.forEach((item) => qs.append(key, item));
      return;
    }
    if (value !== undefined && value !== null && value !== "") {
      qs.set(key, value);
    }
  });
  return qs.toString();
}

export const api = {
  /**
   * POST /chat — note the backend expects the message as a raw
   * text/plain body, not JSON; every other option is a query param.
   * Pass `request_id` in params to make the generation cancellable.
   */
  async sendMessage(message, params, { signal } = {}) {
    const qs = buildQuery(params);
    return request(`/chat?${qs}`, {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: message,
      signal,
    });
  },

  /** POST /chat/stop — asks the backend to stop the generation for this request id. */
  async stopGeneration(requestId) {
    return request(`/chat/stop?${buildQuery({ request_id: requestId })}`, {
      method: "POST",
    });
  },

  async extractFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    return request(`/files/extract`, { method: "POST", body: formData });
  },

  async listConversations({ limit = 20, offset = 0 } = {}) {
    return request(`/history?${buildQuery({ limit, offset })}`);
  },

  async searchConversations(query, { limit = 20 } = {}) {
    return request(`/history?${buildQuery({ q: query, limit })}`);
  },

  async getConversation(id) {
    return request(`/history/${id}`);
  },

  async deleteConversation(id) {
    return request(`/history/${id}`, { method: "DELETE" });
  },

  async ingestDocument(file) {
    const formData = new FormData();
    formData.append("file", file);
    return request(`/rag/ingest`, { method: "POST", body: formData });
  },

  async listDocuments() {
    return request(`/rag/documents`);
  },

  async deleteDocument(id) {
    return request(`/rag/documents/${id}`, { method: "DELETE" });
  },

  async summarize(text, maxLength = 150) {
    return request(`/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, max_length: maxLength }),
    });
  },

  async summarizeByQuery(text, query, maxLength = 150) {
    return request(`/summarize/by-query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, query, max_length: maxLength }),
    });
  },
};