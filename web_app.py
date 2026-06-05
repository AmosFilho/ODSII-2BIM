from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import shutil

from agent import SESSION_ID, agent, knowledge
from agent import build_history_text, build_travel_prompt, buscar_contexto
from agent import load_conversation_history, save_conversation_history

from wiki_engine import (
    list_wiki_pages,
    get_page_content,
    search_wiki,
    find_related_pages,
    build_wiki_index_data,
    render_markdown_to_html,
)

app = FastAPI(title="AmazonIA Travel")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AmazonIA Travel</title>
  <style>
    :root {
      --forest: #164b35;
      --river: #126c78;
      --sun: #f2b84b;
      --leaf: #dcebd2;
      --paper: #fbfaf5;
      --ink: #17201b;
      --muted: #5e6a63;
      --line: #cdd8ce;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: var(--paper);
      color: var(--ink);
    }

    /* ── Tab navigation ──────────────────────────────── */
    .tab-bar {
      display: flex;
      background: var(--forest);
      border-bottom: 2px solid rgba(255,255,255,0.1);
      padding: 0 28px;
    }

    .tab-btn {
      padding: 14px 24px;
      color: rgba(255,255,255,0.65);
      background: none;
      border: none;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 3px solid transparent;
      transition: color 0.2s, border-color 0.2s;
    }

    .tab-btn:hover { color: #fff; }

    .tab-btn.active {
      color: var(--sun);
      border-bottom-color: var(--sun);
    }

    .panel { display: none; }
    .panel.active { display: flex; }

    /* ── Chat layout ─────────────────────────────────── */
    .shell {
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .chat-layout {
      flex: 1;
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
    }

    .sidebar {
      background: var(--forest);
      color: #fff;
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      font-size: 22px;
    }

    .mark {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      background: var(--sun);
      color: var(--forest);
      display: grid;
      place-items: center;
      font-weight: 800;
    }

    .sidebar p {
      margin: 0;
      line-height: 1.5;
      color: #e8f1e9;
    }

    .quick {
      display: grid;
      gap: 10px;
    }

    .quick button {
      width: 100%;
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(255,255,255,0.08);
      color: #fff;
      border-radius: 8px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      font-size: 14px;
    }

    .main {
      padding: 28px;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0 0 6px;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      max-width: 720px;
    }

    .badge {
      border: 1px solid var(--line);
      color: var(--forest);
      background: var(--leaf);
      border-radius: 8px;
      padding: 8px 10px;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 700;
    }

    .chat-window {
      flex: 1;
      min-height: 420px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow-y: auto;
    }

    .message {
      margin-bottom: 14px;
      display: flex;
    }

    .message.user { justify-content: flex-end; }

    .bubble {
      max-width: min(760px, 86%);
      padding: 12px 14px;
      border-radius: 8px;
      white-space: pre-wrap;
      line-height: 1.45;
    }

    .message.user .bubble {
      background: var(--river);
      color: #fff;
    }

    .message.assistant .bubble {
      background: var(--leaf);
      color: var(--ink);
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      margin-top: 14px;
    }

    input[type="text"] {
      min-width: 0;
      padding: 13px 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
      font-size: 16px;
      background: #fff;
    }

    button.action {
      border: none;
      border-radius: 8px;
      background: var(--forest);
      color: #fff;
      padding: 12px 16px;
      cursor: pointer;
      font-size: 15px;
      font-weight: 700;
    }

    button.secondary {
      background: var(--sun);
      color: var(--forest);
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .footer {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }

    .upload-section {
      background: var(--leaf);
      border: 2px dashed var(--forest);
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
      margin-bottom: 14px;
    }

    .upload-section:hover {
      background: rgba(22, 75, 53, 0.1);
      border-color: var(--river);
    }

    .upload-section.dragover {
      background: rgba(18, 108, 120, 0.15);
      border-color: var(--sun);
    }

    .upload-section input[type="file"] {
      display: none;
    }

    .upload-section p {
      margin: 0;
      color: var(--forest);
      font-weight: 600;
    }

    .upload-section small {
      color: var(--muted);
      display: block;
      margin-top: 8px;
    }

    .upload-feedback {
      margin-top: 10px;
      padding: 10px;
      border-radius: 6px;
      font-size: 13px;
      display: none;
    }

    .upload-feedback.success {
      background: rgba(22, 75, 53, 0.2);
      color: var(--forest);
      display: block;
    }

    .upload-feedback.error {
      background: rgba(220, 53, 69, 0.2);
      color: #dc3545;
      display: block;
    }

    /* ── Wiki layout ─────────────────────────────────── */
    .wiki-layout {
      flex: 1;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: 0;
    }

    .wiki-sidebar {
      background: var(--leaf);
      border-right: 1px solid var(--line);
      padding: 20px;
      overflow-y: auto;
    }

    .wiki-sidebar h3 {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--forest);
    }

    .wiki-search-box {
      width: 100%;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 14px;
      margin-bottom: 16px;
      background: #fff;
    }

    .wiki-category {
      margin-bottom: 16px;
    }

    .wiki-category-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
      letter-spacing: 0.3px;
    }

    .wiki-page-link {
      display: block;
      padding: 6px 8px;
      color: var(--ink);
      text-decoration: none;
      font-size: 14px;
      border-radius: 4px;
      cursor: pointer;
      transition: background 0.15s;
    }

    .wiki-page-link:hover {
      background: rgba(22, 75, 53, 0.08);
    }

    .wiki-page-link.active {
      background: var(--forest);
      color: #fff;
    }

    .wiki-main {
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }

    .wiki-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }

    .wiki-toolbar-title {
      font-weight: 700;
      font-size: 18px;
      color: var(--forest);
    }

    .wiki-toolbar-actions {
      display: flex;
      gap: 8px;
    }

    .wiki-toolbar-actions button {
      padding: 8px 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      color: var(--forest);
      transition: background 0.15s;
    }

    .wiki-toolbar-actions button:hover {
      background: var(--leaf);
    }

    .wiki-content-area {
      flex: 1;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      overflow: hidden;
    }

    .wiki-body {
      padding: 28px;
      overflow-y: auto;
    }

    .wiki-body h1 { font-size: 26px; margin-top: 0; }
    .wiki-body h2 { font-size: 20px; color: var(--forest); border-bottom: 1px solid var(--line); padding-bottom: 6px; }
    .wiki-body h3 { font-size: 16px; color: var(--river); }
    .wiki-body p { line-height: 1.6; }
    .wiki-body ul, .wiki-body ol { line-height: 1.8; }
    .wiki-body a { color: var(--river); }
    .wiki-body code {
      background: var(--leaf);
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 14px;
    }
    .wiki-body pre {
      background: #1a2e23;
      color: #dcebd2;
      padding: 14px;
      border-radius: 6px;
      overflow-x: auto;
    }
    .wiki-body pre code { background: none; padding: 0; color: inherit; }
    .wiki-body table {
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0;
    }
    .wiki-body th, .wiki-body td {
      border: 1px solid var(--line);
      padding: 8px 12px;
      text-align: left;
    }
    .wiki-body th { background: var(--leaf); }

    .wiki-toc {
      padding: 20px;
      border-left: 1px solid var(--line);
      overflow-y: auto;
      background: var(--paper);
    }

    .wiki-toc h4 {
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
    }

    .wiki-toc a {
      display: block;
      padding: 4px 0;
      font-size: 13px;
      color: var(--river);
      text-decoration: none;
      line-height: 1.4;
    }

    .wiki-toc a:hover { text-decoration: underline; }
    .wiki-toc .toc-h2 { padding-left: 0; }
    .wiki-toc .toc-h3 { padding-left: 12px; font-size: 12px; }

    .wiki-empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: var(--muted);
      text-align: center;
      padding: 40px;
    }

    .wiki-empty h2 { color: var(--forest); margin-bottom: 8px; }

    .wiki-related {
      margin-top: 28px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }

    .wiki-related h3 { font-size: 16px; color: var(--forest); }

    .wiki-related a {
      display: block;
      padding: 4px 0;
      color: var(--river);
      font-size: 14px;
    }

    @media (max-width: 820px) {
      .chat-layout { grid-template-columns: 1fr; }
      .wiki-layout { grid-template-columns: 1fr; }
      .wiki-sidebar { display: none; }
      .wiki-content-area { grid-template-columns: 1fr; }
      .wiki-toc { display: none; }
      .sidebar { padding: 20px; }
      .main { padding: 20px; }
      header { flex-direction: column; }
      .badge { white-space: normal; }
      .controls { grid-template-columns: 1fr; }
      .bubble { max-width: 100%; }
    }
  </style>
</head>
<body>
  <nav class="tab-bar">
    <button class="tab-btn active" data-tab="chat">💬 Chat</button>
    <button class="tab-btn" data-tab="wiki">📚 Wiki</button>
  </nav>

  <!-- ── Chat Panel ─────────────────────────────────────────────── -->
  <div id="chat-panel" class="panel active">
    <div class="chat-layout">
      <aside class="sidebar">
        <div class="brand">
          <div class="mark">AM</div>
          <div>AmazonIA Travel</div>
        </div>
        <p>IA vertical com RAG para consultar uma wiki de turismo, montar roteiros e apoiar decisões de viagem no Amazonas.</p>
        <div class="quick">
          <button data-prompt="Monte um roteiro de 5 dias em Manaus e arredores para primeira viagem.">Roteiro de 5 dias</button>
          <button data-prompt="Qual a melhor epoca para ver rios, floresta e comunidades ribeirinhas?">Melhor epoca</button>
          <button data-prompt="Quais cuidados devo ter para turismo de selva no Amazonas?">Cuidados na selva</button>
        </div>
      </aside>

      <main class="main">
        <header>
          <div>
            <h1>Consultor inteligente para viagens ao Amazonas</h1>
            <p class="subtitle">Pergunte sobre destinos, logistica, epocas do ano, experiencias culturais, seguranca e planejamento responsavel.</p>
          </div>
          <div class="badge">Base local + consulta atual quando necessario</div>
        </header>

        <div id="chat" class="chat-window"></div>

        <div class="upload-section" id="uploadZone">
          <p>📄 Arrastar arquivos aqui ou clicar para selecionar</p>
          <small>PDF, TXT ou MD (máx. 10 MB)</small>
          <input type="file" id="fileInput" accept=".pdf,.txt,.md" />
          <div class="upload-feedback" id="uploadFeedback"></div>
        </div>

        <div class="controls">
          <input id="message" type="text" placeholder="Ex.: Quero um roteiro com natureza, cultura e pouco deslocamento..." autocomplete="off" />
          <button id="voiceBtn" class="action secondary">Voz</button>
          <button id="sendBtn" class="action">Enviar</button>
        </div>
        <div class="footer">A resposta usa a wiki local como fonte principal. Precos, horarios e disponibilidade devem ser verificados em fonte atual.</div>
      </main>
    </div>
  </div>

  <!-- ── Wiki Panel ─────────────────────────────────────────────── -->
  <div id="wiki-panel" class="panel">
    <div class="wiki-layout">
      <aside class="wiki-sidebar" id="wikiSidebar">
        <h3>📚 Wiki</h3>
        <input type="text" class="wiki-search-box" id="wikiSearch" placeholder="Buscar páginas..." />
        <div id="wikiTree"></div>
      </aside>

      <div class="wiki-main">
        <div class="wiki-toolbar" id="wikiToolbar" style="display:none;">
          <span class="wiki-toolbar-title" id="wikiPageTitle"></span>
          <div class="wiki-toolbar-actions">
            <button id="wikiAskBtn">🤖 Perguntar sobre esta página</button>
          </div>
        </div>

        <div class="wiki-content-area">
          <div class="wiki-body" id="wikiBody">
            <div class="wiki-empty">
              <h2>📚 Wiki do Amazonas</h2>
              <p>Selecione uma página na sidebar para navegar pela base de conhecimento.<br>Use a busca para encontrar tópicos específicos.</p>
            </div>
          </div>
          <aside class="wiki-toc" id="wikiToc" style="display:none;">
            <h4>Nesta página</h4>
            <div id="wikiTocContent"></div>
          </aside>
        </div>
      </div>
    </div>
  </div>

  <script>
    // ── Tab switching ──────────────────────────────────────────
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.panel');

    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab + '-panel').classList.add('active');
      });
    });

    // ── Chat ───────────────────────────────────────────────────
    const chat = document.getElementById('chat');
    const messageInput = document.getElementById('message');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');

    const addMessage = (role, text) => {
      const container = document.createElement('div');
      container.className = `message ${role}`;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      container.appendChild(bubble);
      chat.appendChild(container);
      chat.scrollTop = chat.scrollHeight;
    };

    addMessage('assistant', 'Ola. Sou a AmazonIA Travel. Posso montar roteiros, comparar epocas, sugerir destinos e apontar cuidados para viajar pelo Amazonas.');

    const sendMessage = async (rawText) => {
      const text = rawText || messageInput.value.trim();
      if (!text) return;
      addMessage('user', text);
      messageInput.value = '';
      sendBtn.disabled = true;
      voiceBtn.disabled = true;

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        const data = await response.json();
        addMessage('assistant', data.answer || 'Nao foi possivel obter resposta.');
      } catch (error) {
        addMessage('assistant', 'Erro ao chamar o servidor: ' + error.message);
      } finally {
        sendBtn.disabled = false;
        voiceBtn.disabled = false;
      }
    };

    document.querySelectorAll('[data-prompt]').forEach((button) => {
      button.addEventListener('click', () => sendMessage(button.dataset.prompt));
    });

    sendBtn.addEventListener('click', () => sendMessage());
    messageInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
      }
    });

    let recognition = null;
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRecognition();
      recognition.lang = 'pt-BR';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.addEventListener('result', (event) => {
        const transcript = event.results[0][0].transcript;
        messageInput.value = transcript;
        sendMessage(transcript);
      });

      recognition.addEventListener('end', () => {
        voiceBtn.textContent = 'Voz';
        voiceBtn.disabled = false;
      });
    } else {
      voiceBtn.disabled = true;
      voiceBtn.textContent = 'Sem voz';
    }

    voiceBtn.addEventListener('click', () => {
      if (!recognition) return;
      voiceBtn.disabled = true;
      voiceBtn.textContent = 'Ouvindo...';
      recognition.start();
    });

    // Upload de arquivos
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const uploadFeedback = document.getElementById('uploadFeedback');

    const showFeedback = (message, isSuccess = true) => {
      uploadFeedback.className = `upload-feedback ${isSuccess ? 'success' : 'error'}`;
      uploadFeedback.textContent = message;
      setTimeout(() => { uploadFeedback.className = 'upload-feedback'; }, 5000);
    };

    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
      handleFiles(e.target.files);
    });

    const handleFiles = async (files) => {
      for (let file of files) {
        if (!['application/pdf', 'text/plain', 'text/markdown'].includes(file.type) &&
            !file.name.endsWith('.txt') && !file.name.endsWith('.md') && !file.name.endsWith('.pdf')) {
          showFeedback('Apenas PDF, TXT e MD são aceitos.', false);
          continue;
        }
        if (file.size > 10 * 1024 * 1024) {
          showFeedback('Arquivo muito grande (máx. 10 MB)', false);
          continue;
        }
        await uploadFile(file);
      }
    };

    const uploadFile = async (file) => {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (response.ok) {
          showFeedback(`✓ "${file.name}" adicionado à base de conhecimento!`, true);
        } else {
          showFeedback(`Erro: ${data.detail || 'Falha ao enviar arquivo'}`, false);
        }
      } catch (error) {
        showFeedback('Erro ao enviar arquivo: ' + error.message, false);
      }
    };

    // ── Wiki ───────────────────────────────────────────────────
    const wikiTree = document.getElementById('wikiTree');
    const wikiBody = document.getElementById('wikiBody');
    const wikiToolbar = document.getElementById('wikiToolbar');
    const wikiPageTitle = document.getElementById('wikiPageTitle');
    const wikiToc = document.getElementById('wikiToc');
    const wikiTocContent = document.getElementById('wikiTocContent');
    const wikiSearch = document.getElementById('wikiSearch');
    const wikiAskBtn = document.getElementById('wikiAskBtn');

    let currentWikiPage = null;
    let wikiData = null;

    // Load wiki tree
    const loadWikiTree = async () => {
      try {
        const res = await fetch('/wiki/');
        wikiData = await res.json();
        renderWikiTree(wikiData);
      } catch (e) {
        wikiTree.innerHTML = '<p style="color:var(--muted);font-size:13px;">Erro ao carregar wiki.</p>';
      }
    };

    const renderWikiTree = (data) => {
      let html = '';
      for (const [cat, pages] of Object.entries(data)) {
        html += `<div class="wiki-category">`;
        html += `<div class="wiki-category-title">${cat}</div>`;
        for (const page of pages) {
          html += `<a class="wiki-page-link" data-cat="${cat}" data-slug="${page.slug}" data-file="${page.file}">${page.title}</a>`;
        }
        html += `</div>`;
      }
      wikiTree.innerHTML = html;

      // Bind click events
      wikiTree.querySelectorAll('.wiki-page-link').forEach(link => {
        link.addEventListener('click', () => openWikiPage(link.dataset.cat, link.dataset.slug, link.dataset.file));
      });
    };

    const openWikiPage = async (cat, slug, file) => {
      // Update active state
      wikiTree.querySelectorAll('.wiki-page-link').forEach(l => l.classList.remove('active'));
      const activeLink = wikiTree.querySelector(`[data-slug="${slug}"]`);
      if (activeLink) activeLink.classList.add('active');

      try {
        const res = await fetch(`/wiki/${cat}/${slug}`);
        const page = await res.json();

        currentWikiPage = file;
        wikiPageTitle.textContent = page.title;
        wikiToolbar.style.display = 'flex';
        wikiBody.innerHTML = page.html;

        // Render TOC
        if (page.toc_items && page.toc_items.length > 0) {
          wikiToc.style.display = 'block';
          let tocHtml = '';
          for (const item of page.toc_items) {
            const cls = item.level === 2 ? 'toc-h2' : 'toc-h3';
            tocHtml += `<a href="#${item.anchor}" class="${cls}">${item.title}</a>`;
          }
          wikiTocContent.innerHTML = tocHtml;
        } else {
          wikiToc.style.display = 'none';
        }

        // Load related pages
        await loadRelatedPages(cat, slug);

        // Bind clicks on related links
        wikiBody.querySelectorAll('.wiki-related .wiki-page-link').forEach(link => {
          link.addEventListener('click', (e) => {
            e.preventDefault();
            openWikiPage(link.dataset.cat, link.dataset.slug, link.dataset.file);
          });
        });

        // Scroll to top
        wikiBody.scrollTop = 0;
      } catch (e) {
        wikiBody.innerHTML = '<div class="wiki-empty"><p>Erro ao carregar página.</p></div>';
      }
    };

    const loadRelatedPages = async (cat, slug) => {
      try {
        const res = await fetch(`/wiki/related/${cat}/${slug}`);
        const data = await res.json();
        if (data.related && data.related.length > 0) {
          let html = '<div class="wiki-related"><h3>Páginas relacionadas</h3>';
          for (const r of data.related) {
            html += `<a class="wiki-page-link" data-cat="${r.category}" data-slug="${r.slug}" data-file="${r.file}">${r.title}</a>`;
          }
          html += '</div>';
          wikiBody.insertAdjacentHTML('beforeend', html);
        }
      } catch (e) { /* ignore */ }
    };

    // Wiki search
    let searchTimeout = null;
    wikiSearch.addEventListener('input', () => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(async () => {
        const q = wikiSearch.value.trim();
        if (!q) {
          renderWikiTree(wikiData);
          return;
        }
        try {
          const res = await fetch(`/wiki/search?q=${encodeURIComponent(q)}`);
          const data = await res.json();
          if (data.results.length === 0) {
            wikiTree.innerHTML = '<p style="color:var(--muted);font-size:13px;">Nenhum resultado.</p>';
          } else {
            let html = '<div class="wiki-category"><div class="wiki-category-title">Resultados</div>';
            for (const r of data.results) {
              html += `<a class="wiki-page-link" data-cat="${r.category}" data-slug="${r.slug}" data-file="${r.file}">${r.title}</a>`;
            }
            html += '</div>';
            wikiTree.innerHTML = html;
            wikiTree.querySelectorAll('.wiki-page-link').forEach(link => {
              link.addEventListener('click', () => openWikiPage(link.dataset.cat, link.dataset.slug, link.dataset.file));
            });
          }
        } catch (e) { /* ignore */ }
      }, 300);
    });

    // Ask about page
    wikiAskBtn.addEventListener('click', () => {
      if (!currentWikiPage) return;
      // Switch to chat tab and pre-fill
      tabBtns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      tabBtns[0].classList.add('active');
      document.getElementById('chat-panel').classList.add('active');
      messageInput.value = `Sobre a página "${wikiPageTitle.textContent}": `;
      messageInput.focus();
    });

    // Init
    loadWikiTree();
  </script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"answer": "Por favor, envie uma pergunta valida."}, status_code=400)

    conversation_history = load_conversation_history()
    history_text = build_history_text(conversation_history)
    contexto = buscar_contexto(message)
    prompt = build_travel_prompt(message, contexto, history_text)

    result = agent.run(
        prompt,
        session_id=SESSION_ID,
        add_history_to_context=True,
        add_session_state_to_context=True,
        stream=False,
    )

    answer = result.content if result.content is not None else ""
    conversation_history.append({"role": "user", "content": message})
    conversation_history.append({"role": "assistant", "content": answer})
    save_conversation_history(conversation_history)

    return JSONResponse({"answer": answer})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint para receber arquivos (PDF, TXT, MD) e adicionar à base de conhecimento.
    """
    # Validar extensão
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"detail": "Apenas arquivos PDF, TXT e MD são aceitos."},
            status_code=400,
        )
    
    # Validar tamanho (máx 10 MB)
    max_size = 10 * 1024 * 1024
    file_contents = await file.read()
    if len(file_contents) > max_size:
        return JSONResponse(
            {"detail": "Arquivo muito grande (máximo 10 MB)."},
            status_code=413,
        )
    
    try:
        # Salvar arquivo temporariamente na pasta raw_sources
        raw_sources_dir = Path("raw_sources")
        raw_sources_dir.mkdir(exist_ok=True)
        
        temp_file_path = raw_sources_dir / file.filename
        with open(temp_file_path, "wb") as f:
            f.write(file_contents)
        
        # Adicionar à base de conhecimento
        knowledge.add_content(path=str(temp_file_path))
        
        return JSONResponse(
            {
                "status": "success",
                "message": f"Arquivo '{file.filename}' adicionado com sucesso à base de conhecimento.",
                "file_path": str(temp_file_path),
            },
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(
            {"detail": f"Erro ao processar arquivo: {str(e)}"},
            status_code=500,
        )


# ── Wiki endpoints ─────────────────────────────────────────────────────────────

@app.get("/wiki/")
def wiki_tree() -> JSONResponse:
    """Retorna árvore de categorias e páginas da wiki."""
    tree = list_wiki_pages()
    return JSONResponse(tree)


@app.get("/wiki/index")
def wiki_index() -> JSONResponse:
    """Retorna índice completo da wiki como JSON."""
    data = build_wiki_index_data()
    return JSONResponse(data)


@app.get("/wiki/search")
def wiki_search(q: str = "") -> JSONResponse:
    """Busca páginas da wiki por título ou conteúdo."""
    if not q.strip():
        return JSONResponse({"results": [], "query": q})
    results = search_wiki(q)
    return JSONResponse({"results": results, "query": q})


@app.get("/wiki/{category}/{slug}")
def wiki_page(category: str, slug: str) -> JSONResponse:
    """Retorna conteúdo renderizado de uma página da wiki."""
    page = get_page_content(category, slug)
    if page is None:
        return JSONResponse({"detail": "Página não encontrada."}, status_code=404)
    return JSONResponse(page)


@app.get("/wiki/related/{category}/{slug}")
def wiki_related(category: str, slug: str) -> JSONResponse:
    """Retorna páginas relacionadas a uma dada página."""
    related = find_related_pages(category, slug)
    return JSONResponse({"related": related})


class WikiAskRequest(BaseModel):
    page: str = ""
    question: str = ""


@app.post("/wiki/ask")
def wiki_ask(req: WikiAskRequest) -> JSONResponse:
    """
    Pergunta sobre uma página específica da wiki.
    Usa o conteúdo da página como contexto prioritário para o LLM.
    """
    question = req.question.strip()
    if not question:
        return JSONResponse({"answer": "Por favor, envie uma pergunta válida."}, status_code=400)

    # Busca contexto da página específica se informada
    page_context = ""
    if req.page:
        parts = req.page.split("/", 1)
        if len(parts) == 2:
            page_data = get_page_content(parts[0], parts[1])
            if page_data:
                page_context = f"\n\n--- Conteúdo da página '{page_data['title']}' ---\n{page_data['markdown']}\n--- Fim da página ---\n"

    # Busca contexto adicional via RAG
    rag_context = buscar_contexto(question)

    # Combina contextos
    combined_context = ""
    if page_context:
        combined_context += page_context
    if rag_context:
        combined_context += f"\n\n--- Contexto adicional da base ---\n{rag_context}"

    conversation_history = load_conversation_history()
    history_text = build_history_text(conversation_history)

    prompt = build_travel_prompt(question, combined_context, history_text)

    result = agent.run(
        prompt,
        session_id=SESSION_ID,
        add_history_to_context=True,
        add_session_state_to_context=True,
        stream=False,
    )

    answer = result.content if result.content is not None else ""
    conversation_history.append({"role": "user", "content": f"[Wiki] {question}"})
    conversation_history.append({"role": "assistant", "content": answer})
    save_conversation_history(conversation_history)

    return JSONResponse({"answer": answer, "page": req.page})


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web_app:app", host="127.0.0.1", port=port, reload=True)
