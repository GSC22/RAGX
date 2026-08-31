// =============================================================
// RAG Knowledge Assistant -- frontend application logic.
//
// Plain JavaScript, no framework and no build step: this file is
// served directly by FastAPI's StaticFiles, so "view source" shows
// exactly what runs. State lives in a handful of top-level
// variables (documents, activeDocumentId, thread history) rather
// than a framework store -- appropriate for a single-page tool this
// size, and easy to explain line by line in an interview.
// =============================================================

const API = ""; // same-origin: frontend is served by the same FastAPI app

// ---- state -----------------------------------------------------------

let documents = [];           // [{document_id, filename, status, ...}]
let activeDocumentId = null;
let threads = {};             // document_id -> [{question, answer, sources, notFound}]

// ---- element references -----------------------------------------------

const el = {
  fileInput: document.getElementById("fileInput"),
  docList: document.getElementById("docList"),
  cfgEmbedding: document.getElementById("cfgEmbedding"),
  cfgVectorStore: document.getElementById("cfgVectorStore"),
  cfgLLM: document.getElementById("cfgLLM"),
  chunkSize: document.getElementById("chunkSize"),
  chunkOverlap: document.getElementById("chunkOverlap"),
  topK: document.getElementById("topK"),
  healthStatus: document.getElementById("healthStatus"),
  healthDot: document.getElementById("healthDot"),
  healthText: document.getElementById("healthText"),
  emptyState: document.getElementById("emptyState"),
  activeState: document.getElementById("activeState"),
  docCard: document.getElementById("docCard"),
  thread: document.getElementById("thread"),
  composer: document.getElementById("composer"),
  questionInput: document.getElementById("questionInput"),
  askButton: document.getElementById("askButton"),
};

// ---- API helpers ------------------------------------------------------

async function apiGet(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error((await safeDetail(res)) || `Request failed (${res.status})`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await safeDetail(res)) || `Request failed (${res.status})`);
  return res.json();
}

async function apiUpload(path, file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(API + path, { method: "POST", body: formData });
  if (!res.ok) throw new Error((await safeDetail(res)) || `Upload failed (${res.status})`);
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(API + path, { method: "DELETE" });
  if (!res.ok && res.status !== 204) {
    throw new Error((await safeDetail(res)) || `Delete failed (${res.status})`);
  }
}

async function safeDetail(res) {
  try {
    const data = await res.json();
    return data.detail;
  } catch {
    return null;
  }
}

// ---- health / pipeline info --------------------------------------------

async function loadHealth() {
  try {
    const health = await apiGet("/health");
    el.cfgEmbedding.textContent = health.embedding_model;
    el.cfgVectorStore.textContent = health.vector_store;
    el.cfgLLM.textContent = health.llm_provider;

    if (health.groq_configured) {
      el.healthStatus.className = "health health--ok";
      el.healthText.textContent = "API connected";
    } else {
      el.healthStatus.className = "health health--warn";
      el.healthText.textContent = "GROQ_API_KEY not set -- add it to backend/.env";
    }
  } catch {
    el.healthStatus.className = "health";
    el.healthText.textContent = "Cannot reach the backend";
  }
}

// ---- document list ------------------------------------------------------

async function loadDocuments() {
  const data = await apiGet("/documents");
  documents = data.documents;
  renderDocList();
}

function renderDocList() {
  el.docList.innerHTML = "";

  if (documents.length === 0) {
    const li = document.createElement("li");
    li.className = "doc-list__empty";
    li.textContent = "No documents yet.";
    el.docList.appendChild(li);
    return;
  }

  for (const doc of documents) {
    const li = document.createElement("li");
    li.className = `doc-item doc-item--${doc.status}` + (doc.document_id === activeDocumentId ? " doc-item--active" : "");
    li.setAttribute("role", "button");
    li.tabIndex = 0;

    const statusDot = document.createElement("span");
    statusDot.className = "doc-item__status";

    const name = document.createElement("span");
    name.className = "doc-item__name";
    name.textContent = doc.filename;
    name.title = doc.filename;

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "doc-item__delete";
    deleteBtn.setAttribute("aria-label", `Delete ${doc.filename}`);
    deleteBtn.textContent = "\u00d7";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleDelete(doc.document_id);
    });

    li.append(statusDot, name, deleteBtn);
    li.addEventListener("click", () => selectDocument(doc.document_id));
    el.docList.appendChild(li);
  }
}

async function selectDocument(documentId) {
  const doc = documents.find((d) => d.document_id === documentId);
  if (!doc) return;

  activeDocumentId = documentId;
  renderDocList();

  if (doc.status !== "processed") {
    // Uploaded but not processed yet (e.g. a page reload happened
    // mid-flow) -- process it now using the current sidebar settings.
    await processDocument(documentId);
    return;
  }

  showActiveState(doc);
}

// ---- upload + process ---------------------------------------------------

el.fileInput.addEventListener("change", async () => {
  const file = el.fileInput.files[0];
  if (!file) return;
  el.fileInput.value = ""; // allow re-selecting the same file later

  try {
    const uploaded = await apiUpload("/documents/upload", file);
    await loadDocuments();
    activeDocumentId = uploaded.document_id;
    renderDocList();
    await processDocument(uploaded.document_id);
  } catch (err) {
    showGlobalError(err.message);
  }
});

async function processDocument(documentId) {
  showProcessingState();

  try {
    const chunkSize = parseInt(el.chunkSize.value, 10);
    const chunkOverlap = parseInt(el.chunkOverlap.value, 10);

    await apiPost("/documents/process", {
      document_id: documentId,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
    });

    await loadDocuments();
    const doc = documents.find((d) => d.document_id === documentId);
    threads[documentId] = threads[documentId] || [];
    showActiveState(doc);
  } catch (err) {
    showGlobalError(err.message);
    activeDocumentId = null;
    el.activeState.hidden = true;
    el.emptyState.hidden = false;
  }
}

function showProcessingState() {
  el.emptyState.hidden = true;
  el.activeState.hidden = true;
  const note = document.createElement("div");
  note.className = "empty-state";
  note.id = "processingNote";
  note.innerHTML = `
    <p class="empty-state__eyebrow">Extracting &middot; Chunking &middot; Embedding &middot; Indexing</p>
    <h2 class="empty-state__heading">Processing document&hellip;</h2>
    <p class="empty-state__body">Building the FAISS index. This can take a few seconds, longer the first time while the embedding model loads.</p>
  `;
  document.querySelector(".workspace").appendChild(note);
}

function clearProcessingState() {
  const note = document.getElementById("processingNote");
  if (note) note.remove();
}

// ---- delete -------------------------------------------------------------

async function handleDelete(documentId) {
  try {
    await apiDelete(`/documents/${documentId}`);
    delete threads[documentId];
    if (activeDocumentId === documentId) {
      activeDocumentId = null;
      el.activeState.hidden = true;
      el.emptyState.hidden = false;
    }
    await loadDocuments();
  } catch (err) {
    showGlobalError(err.message);
  }
}

// ---- active document view -----------------------------------------------

function showActiveState(doc) {
  clearProcessingState();
  el.emptyState.hidden = true;
  el.activeState.hidden = false;

  el.docCard.innerHTML = "";
  const name = document.createElement("span");
  name.className = "doc-card__name";
  name.textContent = doc.filename;

  const stats = document.createElement("span");
  stats.className = "doc-card__stats";
  stats.innerHTML = `
    <span>${doc.page_count ?? "?"} pages</span>
    <span>${doc.chunk_count ?? "?"} chunks</span>
    <span>size ${doc.chunk_size ?? "?"}</span>
    <span>overlap ${doc.chunk_overlap ?? "?"}</span>
  `;

  el.docCard.append(name, stats);
  renderThread();
  el.questionInput.focus();
}

function renderThread() {
  el.thread.innerHTML = "";
  const history = threads[activeDocumentId] || [];

  if (history.length === 0) {
    const hint = document.createElement("p");
    hint.className = "thinking";
    hint.textContent = "Ask a question below to get started.";
    el.thread.appendChild(hint);
    return;
  }

  for (const turn of history) {
    el.thread.appendChild(renderQuestion(turn.question));
    el.thread.appendChild(renderAnswer(turn));
  }
  el.thread.scrollTop = el.thread.scrollHeight;
}

function renderQuestion(text) {
  const div = document.createElement("div");
  div.className = "msg-question";
  div.textContent = text;
  return div;
}

function renderAnswer(turn) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg-answer" + (turn.notFound ? " msg-answer--not-found" : "");

  const textEl = document.createElement("div");
  textEl.className = "msg-answer__text";
  textEl.innerHTML = `<p>${escapeHtml(turn.answer)}</p>`;
  wrapper.appendChild(textEl);

  if (turn.sources && turn.sources.length > 0) {
    const sourcesWrap = document.createElement("div");
    sourcesWrap.className = "sources";

    const label = document.createElement("div");
    label.className = "sources__label";
    label.textContent = `SOURCES (${turn.sources.length})`;
    sourcesWrap.appendChild(label);

    for (const source of turn.sources) {
      sourcesWrap.appendChild(renderSourceCard(source));
    }
    wrapper.appendChild(sourcesWrap);
  }

  return wrapper;
}

function renderSourceCard(source) {
  const card = document.createElement("div");
  card.className = "source-card";

  const page = document.createElement("span");
  page.className = "source-card__page";
  page.textContent = `p. ${source.page_number}`;

  const similarity = document.createElement("div");
  similarity.className = "source-card__similarity";
  similarity.textContent = `similarity ${source.similarity.toFixed(3)}`;

  const text = document.createElement("div");
  text.className = "source-card__text";
  text.textContent = truncate(source.text, 220);

  card.append(page, similarity, text);
  return card;
}

// ---- asking questions -----------------------------------------------

el.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.questionInput.value.trim();
  if (!question || !activeDocumentId) return;

  el.questionInput.value = "";
  el.questionInput.disabled = true;
  el.askButton.disabled = true;

  threads[activeDocumentId] = threads[activeDocumentId] || [];
  renderThreadWithPendingQuestion(question);

  try {
    const topK = parseInt(el.topK.value, 10);
    const result = await apiPost("/query", {
      document_id: activeDocumentId,
      question,
      top_k: topK,
    });

    threads[activeDocumentId].push({
      question,
      answer: result.answer,
      sources: result.sources,
      notFound: result.sources.length === 0 || /couldn.t find enough information/i.test(result.answer),
    });
  } catch (err) {
    threads[activeDocumentId].push({
      question,
      answer: `Something went wrong: ${err.message}`,
      sources: [],
      notFound: true,
    });
  } finally {
    el.questionInput.disabled = false;
    el.askButton.disabled = false;
    renderThread();
    el.questionInput.focus();
  }
});

function renderThreadWithPendingQuestion(question) {
  el.thread.innerHTML = "";
  const history = threads[activeDocumentId] || [];
  for (const turn of history) {
    el.thread.appendChild(renderQuestion(turn.question));
    el.thread.appendChild(renderAnswer(turn));
  }
  el.thread.appendChild(renderQuestion(question));

  const pending = document.createElement("p");
  pending.className = "thinking";
  pending.textContent = "Retrieving relevant chunks and generating an answer\u2026";
  el.thread.appendChild(pending);

  el.thread.scrollTop = el.thread.scrollHeight;
}

// ---- misc helpers -------------------------------------------------------

function showGlobalError(message) {
  clearProcessingState();
  el.activeState.hidden = true;
  el.emptyState.hidden = false;
  el.emptyState.innerHTML = `
    <p class="empty-state__eyebrow">Something went wrong</p>
    <div class="error-banner">${escapeHtml(message)}</div>
  `;
}

function truncate(text, maxLength) {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trim() + "\u2026";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---- boot -------------------------------------------------------------

(async function init() {
  await loadHealth();
  await loadDocuments();
})();
