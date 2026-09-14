// app.js — chat console UI logic. All backend calls are async (fetch),
// the API always returns/accepts JSON except /chat's request body,
// which the backend defines as raw text/plain.

import { api } from "./api.js";

const state = {
  conversationId: null,
  conversations: [],
  contextText: null,
  contextFilename: null,
  sending: false,
};

const el = {
  sidebar: document.getElementById("sidebar"),
  sidebarToggleBtn: document.getElementById("sidebar-toggle-btn"),
  sidebarBackdrop: document.getElementById("sidebar-backdrop"),
  convList: document.getElementById("conversation-list"),
  searchInput: document.getElementById("search-input"),
  newChatBtn: document.getElementById("new-chat-btn"),

  messages: document.getElementById("messages"),
  emptyState: document.getElementById("empty-state"),
  conversationTitle: document.getElementById("conversation-title"),
  deleteConvBtn: document.getElementById("delete-conversation-btn"),

  input: document.getElementById("message-input"),
  sendBtn: document.getElementById("send-btn"),
  attachBtn: document.getElementById("attach-btn"),
  fileInput: document.getElementById("file-input"),
  contextChip: document.getElementById("context-chip"),
  contextChipLabel: document.getElementById("context-chip-label"),
  contextChipRemove: document.getElementById("context-chip-remove"),

  thinkingMode: document.getElementById("thinking-mode"),
  personality: document.getElementById("personality"),
  language: document.getElementById("language"),
  webSearchToggle: document.getElementById("web-search-toggle"),
  ragToggle: document.getElementById("rag-toggle"),
  ragTopKWrap: document.getElementById("rag-top-k-wrap"),
  ragTopK: document.getElementById("rag-top-k"),

  ragDrawerBtn: document.getElementById("rag-toggle-btn"),
  ragDrawer: document.getElementById("rag-drawer"),
  ragDrawerClose: document.getElementById("rag-drawer-close"),
  ragDocList: document.getElementById("rag-doc-list"),
  ragUploadInput: document.getElementById("rag-upload-input"),
  ragUploadStatus: document.getElementById("rag-upload-status"),
  drawerBackdrop: document.getElementById("drawer-backdrop"),
};

// --- helpers ---

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/** Minimal, safe formatting: escape everything first, then re-introduce
 * paragraphs / line breaks / code blocks. No HTML from the model is ever
 * trusted directly. */
function renderMarkdownLite(text) {
  let safe = escapeHtml(text);
  safe = safe.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
  safe = safe.replace(/\n{2,}/g, "</p><p>");
  safe = safe.replace(/\n/g, "<br>");
  return `<p>${safe}</p>`;
}

function autoResizeInput() {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 160) + "px";
}

// --- messages ---

function appendMessage(role, content, { pending = false } = {}) {
  el.emptyState.classList.add("hidden");

  const wrapper = document.createElement("div");
  wrapper.className = `flex ${role === "user" ? "flex-row-reverse" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = `msg-bubble rounded-lg px-4 py-3 text-sm ${
    role === "user" ? "bg-elevated border border-border" : "text-ink"
  }`;

  if (pending) {
    bubble.dataset.pending = "true";
    bubble.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
  } else {
    bubble.innerHTML = renderMarkdownLite(content);
  }

  wrapper.appendChild(bubble);
  el.messages.appendChild(wrapper);
  el.messages.scrollTop = el.messages.scrollHeight;
  return bubble;
}

// --- conversation list / history ---

function renderConversationList(items, { activeId = state.conversationId } = {}) {
  el.convList.innerHTML = "";

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "text-xs text-dim px-3 py-4 text-center";
    empty.textContent = "گفتگویی ثبت نشده است";
    el.convList.appendChild(empty);
    return;
  }

  items.forEach((conv) => {
    const btn = document.createElement("button");
    btn.className = `conv-item w-full text-right rounded-md px-3 py-2 text-sm border border-transparent hover:bg-elevated transition-colors truncate block ${
      conv.id === activeId ? "active" : ""
    }`;
    btn.textContent = conv.title || `گفتگو #${conv.id}`;
    if (conv.snippet) btn.title = conv.snippet;
    btn.addEventListener("click", () => loadConversation(conv.id));
    el.convList.appendChild(btn);
  });
}

async function refreshConversations(query = "") {
  try {
    const data = query ? await api.searchConversations(query) : await api.listConversations();
    state.conversations = data.results || data.conversations || [];
    renderConversationList(state.conversations);
  } catch (err) {
    console.error("Failed to load conversations:", err);
  }
}

async function loadConversation(id) {
  try {
    const conv = await api.getConversation(id);
    state.conversationId = conv.id;

    el.conversationTitle.textContent = conv.title;
    el.deleteConvBtn.classList.remove("hidden");
    el.messages.innerHTML = "";
    el.emptyState.classList.add("hidden");
    conv.messages.forEach((m) => appendMessage(m.role, m.content));

    renderConversationList(state.conversations, { activeId: id });
    closeMobileSidebar();
  } catch (err) {
    console.error("Failed to load conversation:", err);
  }
}

function startNewChat() {
  state.conversationId = null;
  el.messages.innerHTML = "";
  el.emptyState.classList.remove("hidden");
  el.conversationTitle.textContent = "گفتگوی جدید";
  el.deleteConvBtn.classList.add("hidden");
  clearContext();
  renderConversationList(state.conversations, { activeId: null });
  closeMobileSidebar();
  el.input.focus();
}

async function deleteCurrentConversation() {
  if (!state.conversationId) return;
  if (!confirm("این گفتگو حذف شود؟")) return;

  try {
    await api.deleteConversation(state.conversationId);
    await refreshConversations();
    startNewChat();
  } catch (err) {
    alert(`حذف ناموفق بود: ${err.message}`);
  }
}

// --- context (extracted file text passed as `context`) ---

function setContext(filename, content) {
  state.contextText = content;
  state.contextFilename = filename;
  el.contextChipLabel.textContent = filename;
  el.contextChip.classList.remove("hidden");
  el.contextChip.classList.add("flex");
}

function clearContext() {
  state.contextText = null;
  state.contextFilename = null;
  el.contextChip.classList.add("hidden");
  el.contextChip.classList.remove("flex");
  el.fileInput.value = "";
}

async function handleFileAttach(e) {
  const file = e.target.files[0];
  if (!file) return;

  try {
    el.attachBtn.disabled = true;
    const result = await api.extractFile(file);
    setContext(result.filename, result.content);
  } catch (err) {
    alert(`استخراج فایل ناموفق بود: ${err.message}`);
  } finally {
    el.attachBtn.disabled = false;
  }
}

// --- sending a message ---

async function sendMessage() {
  const text = el.input.value.trim();
  if (!text || state.sending) return;

  state.sending = true;
  el.sendBtn.disabled = true;
  el.input.value = "";
  autoResizeInput();

  appendMessage("user", text);
  const pendingBubble = appendMessage("assistant", "", { pending: true });

  const params = {
    thinking_mode: el.thinkingMode.value,
    personality: el.personality.value,
    language: el.language.value,
    web_search: el.webSearchToggle.checked,
    rag: el.ragToggle.checked,
    rag_top_k: el.ragToggle.checked ? el.ragTopK.value : undefined,
    context: state.contextText || undefined,
    conversation_id: state.conversationId || undefined,
  };

  try {
    const result = await api.sendMessage(text, params);

    pendingBubble.innerHTML = renderMarkdownLite(result.response);
    delete pendingBubble.dataset.pending;

    const wasNewConversation = !state.conversationId;
    state.conversationId = result.conversation_id;

    if (wasNewConversation && state.conversationId) {
      el.deleteConvBtn.classList.remove("hidden");
      el.conversationTitle.textContent = text.slice(0, 48);
      await refreshConversations();
      renderConversationList(state.conversations, { activeId: state.conversationId });
    }
  } catch (err) {
    pendingBubble.innerHTML = `<span class="text-danger">خطا: ${escapeHtml(err.message)}</span>`;
    delete pendingBubble.dataset.pending;
  } finally {
    state.sending = false;
    el.sendBtn.disabled = false;
  }
}

// --- RAG documents drawer ---

function openDrawer() {
  el.ragDrawer.classList.add("open");
  el.drawerBackdrop.classList.remove("hidden");
  refreshDocuments();
}

function closeDrawer() {
  el.ragDrawer.classList.remove("open");
  el.drawerBackdrop.classList.add("hidden");
}

async function refreshDocuments() {
  el.ragDocList.innerHTML = `<p class="text-xs text-dim px-3 py-4">در حال بارگذاری...</p>`;

  try {
    const { documents } = await api.listDocuments();

    if (!documents.length) {
      el.ragDocList.innerHTML = `<p class="text-xs text-dim px-3 py-4 text-center">سندی بارگذاری نشده است</p>`;
      return;
    }

    el.ragDocList.innerHTML = "";
    documents.forEach((doc) => {
      const row = document.createElement("div");
      row.className = "flex items-center justify-between gap-2 px-4 py-2.5 border-b border-border";
      row.innerHTML = `
        <div class="min-w-0">
          <p class="text-sm truncate">${escapeHtml(doc.filename)}</p>
          <p class="text-xs text-dim font-mono" dir="ltr">${doc.chunk_count} chunks</p>
        </div>
        <button data-id="${doc.id}" class="doc-delete-btn text-dim hover:text-danger text-xs shrink-0">حذف</button>
      `;
      el.ragDocList.appendChild(row);
    });

    el.ragDocList.querySelectorAll(".doc-delete-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api.deleteDocument(Number(btn.dataset.id));
          refreshDocuments();
        } catch (err) {
          alert(`حذف سند ناموفق بود: ${err.message}`);
        }
      });
    });
  } catch (err) {
    el.ragDocList.innerHTML = `<p class="text-xs text-danger px-4 py-4">${escapeHtml(err.message)}</p>`;
  }
}

async function handleDocumentUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  el.ragUploadStatus.textContent = "در حال بارگذاری...";
  try {
    const result = await api.ingestDocument(file);
    el.ragUploadStatus.textContent = `${result.chunks_created} تکه ساخته شد`;
    refreshDocuments();
  } catch (err) {
    el.ragUploadStatus.textContent = `خطا: ${err.message}`;
  } finally {
    e.target.value = "";
  }
}

// --- mobile sidebar ---

function openMobileSidebar() {
  el.sidebar.classList.remove("hidden");
  el.sidebar.classList.add("mobile-open");
  el.sidebarBackdrop.classList.remove("hidden");
}

function closeMobileSidebar() {
  if (window.innerWidth >= 768) return;
  el.sidebar.classList.add("hidden");
  el.sidebar.classList.remove("mobile-open");
  el.sidebarBackdrop.classList.add("hidden");
}

// --- events ---

el.newChatBtn.addEventListener("click", startNewChat);
el.deleteConvBtn.addEventListener("click", deleteCurrentConversation);

el.sendBtn.addEventListener("click", sendMessage);
el.input.addEventListener("input", autoResizeInput);
el.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

el.attachBtn.addEventListener("click", () => el.fileInput.click());
el.fileInput.addEventListener("change", handleFileAttach);
el.contextChipRemove.addEventListener("click", clearContext);

el.ragToggle.addEventListener("change", () => {
  const show = el.ragToggle.checked;
  el.ragTopKWrap.classList.toggle("hidden", !show);
  el.ragTopKWrap.classList.toggle("flex", show);
});

let searchDebounce;
el.searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => refreshConversations(el.searchInput.value.trim()), 300);
});

el.ragDrawerBtn.addEventListener("click", openDrawer);
el.ragDrawerClose.addEventListener("click", closeDrawer);
el.drawerBackdrop.addEventListener("click", closeDrawer);
el.ragUploadInput.addEventListener("change", handleDocumentUpload);

el.sidebarToggleBtn.addEventListener("click", openMobileSidebar);
el.sidebarBackdrop.addEventListener("click", closeMobileSidebar);

// --- init ---

refreshConversations();
