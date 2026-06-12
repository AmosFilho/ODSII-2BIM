from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import json
import uuid
import shutil

from agent import SESSION_ID, agent, knowledge, PREPROCESS_MODEL
from agent import build_history_text, build_travel_prompt, buscar_contexto
from agent import load_conversation_history, save_conversation_history
from agent import safe_agent_run

from wiki_engine import (
    list_wiki_pages,
    get_page_content,
    search_wiki,
    find_related_pages,
    build_wiki_index_data,
    render_markdown_to_html,
    chunk_markdown,
)
import wiki_graph
from agno.vectordb.chroma import SearchType

# ── In-memory task store for SSE upload progress ────────────────────────────
_upload_tasks: dict[str, dict] = {}

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

    /* ── Markdown inside chat bubbles ────────────────────────── */
    .message.assistant .bubble h1,
    .message.assistant .bubble h2,
    .message.assistant .bubble h3,
    .message.assistant .bubble h4 {
      margin: 10px 0 6px;
      line-height: 1.3;
      color: var(--forest);
    }
    .message.assistant .bubble h1 { font-size: 20px; }
    .message.assistant .bubble h2 { font-size: 17px; }
    .message.assistant .bubble h3 { font-size: 15px; }
    .message.assistant .bubble h4 { font-size: 14px; }
    .message.assistant .bubble p {
      margin: 6px 0;
      line-height: 1.5;
    }
    .message.assistant .bubble ul,
    .message.assistant .bubble ol {
      margin: 6px 0;
      padding-left: 22px;
      line-height: 1.6;
    }
    .message.assistant .bubble li { margin: 2px 0; }
    .message.assistant .bubble strong { color: var(--forest); }
    .message.assistant .bubble a { color: var(--river); text-decoration: underline; }
    .message.assistant .bubble code {
      background: rgba(22, 75, 53, 0.12);
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 13px;
      font-family: 'Courier New', monospace;
    }
    .message.assistant .bubble pre {
      background: #1a2e23;
      color: #dcebd2;
      padding: 10px 12px;
      border-radius: 6px;
      overflow-x: auto;
      margin: 8px 0;
    }
    .message.assistant .bubble pre code {
      background: none;
      padding: 0;
      color: inherit;
      font-size: 13px;
    }
    .message.assistant .bubble blockquote {
      border-left: 3px solid var(--forest);
      margin: 8px 0;
      padding: 4px 12px;
      color: var(--muted);
      font-style: italic;
    }
    .message.assistant .bubble table {
      border-collapse: collapse;
      width: 100%;
      margin: 8px 0;
      font-size: 14px;
    }
    .message.assistant .bubble th,
    .message.assistant .bubble td {
      border: 1px solid var(--line);
      padding: 6px 10px;
      text-align: left;
    }
    .message.assistant .bubble th {
      background: rgba(22, 75, 53, 0.1);
      font-weight: 700;
      color: var(--forest);
    }
    .message.assistant .bubble hr {
      border: none;
      border-top: 1px solid var(--line);
      margin: 10px 0;
    }

    /* ── Typing indicator ─────────────────────────────── */
    .typing-indicator {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 12px 14px;
    }

    .typing-indicator .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--muted);
      animation: typing-bounce 1.4s infinite ease-in-out both;
    }

    .typing-indicator .dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator .dot:nth-child(2) { animation-delay: -0.16s; }
    .typing-indicator .dot:nth-child(3) { animation-delay: 0s; }

    @keyframes typing-bounce {
      0%, 80%, 100% {
        transform: scale(0.4);
        opacity: 0.4;
      }
      40% {
        transform: scale(1);
        opacity: 1;
      }
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

    /* ── Upload progress bar ─────────────────────────────── */
    .upload-progress {
      display: none;
      margin-top: 12px;
      background: rgba(22,75,53,0.07);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 13px;
    }
    .upload-progress.active { display: block; }
    .upload-progress-bar-wrap {
      background: var(--line);
      border-radius: 4px;
      height: 6px;
      margin: 8px 0 10px;
      overflow: hidden;
    }
    .upload-progress-bar {
      height: 6px;
      background: linear-gradient(90deg, var(--forest), var(--river));
      border-radius: 4px;
      width: 0%;
      transition: width 0.5s ease;
    }
    .upload-progress-steps { display: grid; gap: 4px; }
    .upload-step {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .upload-step.done { color: var(--forest); font-weight: 600; }
    .upload-step.active { color: var(--river); font-weight: 600; }
    .upload-step .step-icon { font-size: 14px; min-width: 18px; text-align: center; }

    /* ── Source attribution block ────────────────────────── */
    .source-block {
      margin-top: 10px;
      padding: 8px 12px;
      background: rgba(22,75,53,0.08);
      border-left: 3px solid var(--forest);
      border-radius: 0 6px 6px 0;
      font-size: 12px;
      color: var(--muted);
    }
    .source-block strong { color: var(--forest); }

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
          <div class="badge">Base local (Wiki)</div>
        </header>

        <div id="chat" class="chat-window"></div>

        <div class="upload-section" id="uploadZone">
          <p>📄 Arrastar arquivos aqui ou clicar para selecionar</p>
          <small>PDF, TXT ou MD (máx. 10 MB)</small>
          <input type="file" id="fileInput" accept=".pdf,.txt,.md" />
          <div class="upload-feedback" id="uploadFeedback"></div>
        </div>
        <div class="upload-progress" id="uploadProgress">
          <div id="uploadProgressLabel" style="font-weight:600;color:var(--forest);margin-bottom:4px;">Processando...</div>
          <div class="upload-progress-bar-wrap">
            <div class="upload-progress-bar" id="uploadProgressBar"></div>
          </div>
          <div class="upload-progress-steps" id="uploadSteps">
            <div class="upload-step" id="step-read"><span class="step-icon">📂</span> Lendo arquivo</div>
            <div class="upload-step" id="step-preprocess"><span class="step-icon">🤖</span> Pré-processando com LLM</div>
            <div class="upload-step" id="step-embed"><span class="step-icon">🔢</span> Gerando embeddings</div>
            <div class="upload-step" id="step-index"><span class="step-icon">🗄️</span> Indexando no banco vetorial</div>
            <div class="upload-step" id="step-done"><span class="step-icon">✅</span> Concluído</div>
          </div>
        </div>

        <div class="controls">
          <input id="message" type="text" placeholder="Ex.: Quero um roteiro com natureza, cultura e pouco deslocamento..." autocomplete="off" />
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

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    // ── Markdown rendering for chat messages ───────────────────
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        breaks: true,
        gfm: true,
      });
    }

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

    // addMessage is now redefined below with sources support

    const showTyping = () => {
      const container = document.createElement('div');
      container.className = 'message assistant';
      container.id = 'typingIndicator';
      const bubble = document.createElement('div');
      bubble.className = 'bubble typing-indicator';
      bubble.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
      container.appendChild(bubble);
      chat.appendChild(container);
      chat.scrollTop = chat.scrollHeight;
    };

    const hideTyping = () => {
      const el = document.getElementById('typingIndicator');
      if (el) el.remove();
    };

    const addMessage = (role, text, sources) => {
      const container = document.createElement('div');
      container.className = `message ${role}`;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      if (role === 'assistant' && typeof marked !== 'undefined') {
        bubble.innerHTML = marked.parse(text);
        // Append source block if sources available
        if (sources && sources.length > 0) {
          const srcBlock = document.createElement('div');
          srcBlock.className = 'source-block';
          srcBlock.innerHTML = `<strong>📄 Fontes consultadas:</strong> ${sources.map(s => `<code>${s}</code>`).join(', ')}`;
          bubble.appendChild(srcBlock);
        }
      } else {
        bubble.textContent = text;
      }
      container.appendChild(bubble);
      chat.appendChild(container);
      chat.scrollTop = chat.scrollHeight;
    };

    // Override initial greeting
    chat.innerHTML = '';
    addMessage('assistant', 'Ola. Sou a AmazonIA Travel. Posso montar roteiros, comparar epocas, sugerir destinos e apontar cuidados para viajar pelo Amazonas.');

    const sendMessage = async (rawText) => {
      const text = rawText || messageInput.value.trim();
      if (!text) return;
      addMessage('user', text);
      messageInput.value = '';
      sendBtn.disabled = true;
      showTyping();

      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        const data = await response.json();
        hideTyping();
        addMessage('assistant', data.answer || 'Nao foi possivel obter resposta.', data.sources);
      } catch (error) {
        hideTyping();
        addMessage('assistant', 'Erro ao chamar o servidor: ' + error.message);
      } finally {
        sendBtn.disabled = false;
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

    // ── Upload progress helpers ──────────────────────────────────────────
    const STEPS = ['read', 'preprocess', 'embed', 'index', 'done'];
    const STEP_PROGRESS = { read: 10, preprocess: 40, embed: 65, index: 85, done: 100 };
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadProgressLabel = document.getElementById('uploadProgressLabel');

    const resetProgressUI = () => {
      STEPS.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        if (el) { el.classList.remove('active', 'done'); }
      });
      uploadProgressBar.style.width = '0%';
      uploadProgressLabel.textContent = 'Processando...';
      uploadProgress.classList.add('active');
    };

    const applyStep = (step) => {
      const pct = STEP_PROGRESS[step] || 0;
      uploadProgressBar.style.width = pct + '%';
      STEPS.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        if (!el) return;
        if (s === step) {
          el.classList.add('active');
          el.classList.remove('done');
        } else if (STEPS.indexOf(s) < STEPS.indexOf(step)) {
          el.classList.remove('active');
          el.classList.add('done');
        }
      });
      if (step === 'done') {
        uploadProgressLabel.textContent = '✓ Concluído!';
        setTimeout(() => uploadProgress.classList.remove('active'), 3000);
      }
    };

    const uploadFile = async (file) => {
      const formData = new FormData();
      formData.append('file', file);

      resetProgressUI();
      applyStep('read');

      try {
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const data = await response.json();
          uploadProgress.classList.remove('active');
          showFeedback(`Erro: ${data.detail || 'Falha ao enviar arquivo'}`, false);
          return;
        }

        const data = await response.json();
        const taskId = data.task_id;

        if (!taskId) {
          uploadProgress.classList.remove('active');
          showFeedback(`✓ "${file.name}" adicionado!`, true);
          return;
        }

        // Connect to SSE stream
        const evtSource = new EventSource(`/upload/progress/${taskId}`);
        evtSource.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          if (msg.step) applyStep(msg.step);
          if (msg.label) uploadProgressLabel.textContent = msg.label;
          if (msg.done) {
            evtSource.close();
            if (msg.error) {
              uploadProgress.classList.remove('active');
              showFeedback(`Erro: ${msg.error}`, false);
            } else {
              showFeedback(`✓ "${file.name}" adicionado à base de conhecimento!`, true);
            }
          }
        };
        evtSource.onerror = () => {
          evtSource.close();
          uploadProgress.classList.remove('active');
        };
      } catch (error) {
        uploadProgress.classList.remove('active');
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

        // Load related pages only if not already present in the Markdown content (Obsidian integration)
        if (!wikiBody.innerHTML.includes('Páginas Relacionadas')) {
          await loadRelatedPages(cat, slug);
          
          // Bind clicks on dynamically loaded related links
          wikiBody.querySelectorAll('.wiki-related .wiki-page-link').forEach(link => {
            link.addEventListener('click', (e) => {
              e.preventDefault();
              openWikiPage(link.dataset.cat, link.dataset.slug, link.dataset.file);
            });
          });
        }

        // Bind clicks on standard relative links inside the markdown body (Obsidian integration)
        wikiBody.querySelectorAll('a').forEach(link => {
          const href = link.getAttribute('href');
          if (href && href.startsWith('../')) {
            const parts = href.split('/');
            if (parts.length >= 3) {
              const rCat = parts[parts.length - 2];
              const rFile = parts[parts.length - 1];
              const rSlug = rFile.replace('.md', '');
              
              link.addEventListener('click', (e) => {
                e.preventDefault();
                openWikiPage(rCat, rSlug, `${rCat}/${rFile}`);
              });
            }
          }
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
    contexto, fontes = buscar_contexto(message)
    prompt = build_travel_prompt(message, contexto, history_text, fontes)

    answer = safe_agent_run(prompt, session_id=SESSION_ID)
    conversation_history.append({"role": "user", "content": message})
    conversation_history.append({"role": "assistant", "content": answer})
    save_conversation_history(conversation_history)

    return JSONResponse({"answer": answer, "sources": fontes})

@app.get("/upload/progress/{task_id}")
async def upload_progress(task_id: str):
    """
    SSE endpoint — publica eventos de progresso do upload enquanto a tarefa roda.
    Cada evento é um JSON com: step, label, done, error (opcional).
    """
    async def event_generator():
        while True:
            task = _upload_tasks.get(task_id)
            if task is None:
                yield f"data: {json.dumps({'done': True, 'error': 'task not found'})}\n\n"
                break

            events: list[dict] = task.get("events", [])
            last_sent = task.get("last_sent", 0)

            for evt in events[last_sent:]:
                yield f"data: {json.dumps(evt)}\n\n"
                last_sent += 1

            task["last_sent"] = last_sent

            if task.get("finished"):
                # Clean up after sending all events
                _upload_tasks.pop(task_id, None)
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Recebe arquivos (PDF, TXT, MD), converte para Markdown (com LLM para .txt/.pdf),
    compara embeddings com a wiki, atualiza páginas semelhantes e indexa no RAG.
    Retorna task_id imediatamente; progresso exposto via SSE em /upload/progress/{task_id}.
    """
    import re
    import math
    import pypdf
    import ollama
    from agent import vector_db, CHAT_MODEL
    from wiki_engine import _parse_frontmatter

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

    # Criar task e registrar no store
    task_id = str(uuid.uuid4())
    _upload_tasks[task_id] = {"events": [], "finished": False, "last_sent": 0}

    def push(step: str, label: str, done: bool = False, error: str = ""):
        evt: dict = {"step": step, "label": label, "done": done}
        if error:
            evt["error"] = error
        _upload_tasks[task_id]["events"].append(evt)
        if done:
            _upload_tasks[task_id]["finished"] = True

    # Snapshot dos dados necessários antes de lançar background task
    filename = file.filename

    async def process():
        try:
            # ── Step 1: Ler arquivo ───────────────────────────────────────────
            push("read", "📂 Lendo arquivo...")
            raw_sources_dir = Path("raw_sources")
            raw_sources_dir.mkdir(exist_ok=True)
            temp_file_path = raw_sources_dir / filename
            with open(temp_file_path, "wb") as f:
                f.write(file_contents)

            extracted_text = ""
            if file_ext == ".pdf":
                reader = pypdf.PdfReader(temp_file_path)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""
            else:
                extracted_text = file_contents.decode("utf-8", errors="ignore")

            stem = Path(filename).stem
            slug = re.sub(r"[^a-zA-ZÀ-ÿ0-9 _-]", "", stem)
            slug = slug.strip().replace(" ", "-").lower()
            title = stem.replace("-", " ").replace("_", " ").title()

            # ── Step 2: Pré-processamento LLM (apenas .txt e .pdf) ──────────
            if file_ext in (".txt", ".pdf"):
                push("preprocess", "🤖 Pré-processando com LLM...")
                preprocess_prompt = f"""Você é um editor técnico especializado em turismo no Amazonas.
Sua tarefa é transformar o texto bruto abaixo em um documento Markdown bem estruturado.

Regras:
1. Use um título principal H1 (# Título) descritivo.
2. Organize o conteúdo em seções com headers H2 (## Seção) e H3 (### Subção) quando adequado.
3. Use listas com marcadores para enumerações.
4. Use tabelas Markdown para dados comparativos.
5. Preserve todos os fatos e dados do texto original.
6. Não invente informações.
7. Retorne APENAS o Markdown final, sem comentários ou blocos de código extras.

--- TEXTO BRUTO ---
{extracted_text[:8000]}
--- FIM DO TEXTO ---"""
                llm_resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ollama.generate(
                        model=PREPROCESS_MODEL,
                        prompt=preprocess_prompt,
                        options={"temperature": 0.2},
                    ),
                )
                markdown_content = llm_resp["response"].strip()
                # Limpar possivel bloco ```markdown
                if markdown_content.startswith("```markdown"):
                    markdown_content = markdown_content[11:].strip()
                elif markdown_content.startswith("```"):
                    markdown_content = markdown_content[3:].strip()
                if markdown_content.endswith("```"):
                    markdown_content = markdown_content[:-3].strip()
            else:
                # .md já está formatado — pular pré-processamento
                push("preprocess", "⏩ Arquivo .md detectado, pré-processamento ignorado")
                if not extracted_text.strip().startswith("#"):
                    markdown_content = f"# {title}\n\n{extracted_text}"
                else:
                    markdown_content = extracted_text

            # 1. Determinar a categoria com base no frontmatter ou no conteúdo
            from wiki_engine import _parse_frontmatter
            meta, _ = _parse_frontmatter(markdown_content)
            category = meta.get("category", "").lower().strip()

            def guess_category(text: str) -> str:
                text_lower = text.lower()
                if any(w in text_lower for w in ["roteiro", "itinerário", "dia 1", "dia 2", "dia 3"]):
                    return "synthesis"
                if any(w in text_lower for w in ["comparação", "comparativo", "diferença entre"]):
                    return "comparisons"
                if any(w in text_lower for w in ["visão geral", "panorama", "introdução ao turismo"]):
                    return "overviews"
                if any(w in text_lower for w in ["logística", "época", "segurança", "saúde", "sustentabilidade", "vacina"]):
                    return "concepts"
                return "entities"

            if category not in {"entities", "concepts", "comparisons", "overviews", "synthesis"}:
                category = guess_category(markdown_content)

            # Salvar na pasta da categoria correspondente
            wiki_category_dir = Path("wiki") / category
            wiki_category_dir.mkdir(exist_ok=True)
            new_md_path = wiki_category_dir / f"{slug}.md"
            new_md_path.write_text(markdown_content, encoding="utf-8")

            # ── Step 3: Embeddings ────────────────────────────────────────────
            push("embed", "🔢 Gerando embeddings e indexando...")

            # Deleta versão antiga da base de conhecimento (se houver) e adiciona a nova
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vector_db.delete_by_metadata(metadata={"name": new_md_path.name})
            )
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: knowledge.add_content(path=str(new_md_path))
            )

            # ── Step 4: Indexar ───────────────────────────────────────────────
            push("index", "🗄️ Atualizando grafo de relacionamentos...")

            # Alterna temporariamente para busca vetorial pura para obter as distâncias
            orig_search_type = vector_db.search_type
            vector_db.search_type = SearchType.vector

            # Busca os 20 vizinhos mais próximos no ChromaDB usando o conteúdo do novo documento
            raw_neighbors = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vector_db.search(markdown_content, limit=20)
            )

            # Restaura o tipo de busca original
            vector_db.search_type = orig_search_type

            neighbor_entries: dict[str, float] = {}
            for doc in raw_neighbors:
                doc_name = getattr(doc, "name", "")
                if not doc_name or doc_name == new_md_path.name:
                    continue
                if doc_name == "index.md":
                    continue

                # Localizar o caminho relativo correspondente ao arquivo
                doc_file = ""
                wiki_dir_path = Path("wiki")
                for candidate in wiki_dir_path.rglob("*.md"):
                    if candidate.name == doc_name:
                        doc_file = str(candidate.relative_to(wiki_dir_path)).replace("\\", "/")
                        break

                if not doc_file:
                    continue

                # Extrair distância e calcular similaridade cosseno
                distance = doc.meta_data.get("distances", 1.0) if hasattr(doc, "meta_data") and doc.meta_data else 1.0
                similarity = 1.0 - distance

                # Reter apenas conexões com relação relevante (similaridade >= 0.5)
                if similarity >= 0.5:
                    if doc_file not in neighbor_entries or similarity > neighbor_entries[doc_file]:
                        neighbor_entries[doc_file] = round(similarity, 4)

            # Atualiza o grafo de conexões bidirecionais no arquivo .obsidian/graph.json
            page_id = f"{category}/{slug}.md"
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: wiki_graph.update_page_relations(page_id, neighbor_entries)
            )

            # Anexa os links das páginas relacionadas no final do novo Markdown para visualização no gráfico do Obsidian
            if neighbor_entries:
                links_md = "\n\n## Páginas Relacionadas\n"
                for neighbor_file, similarity in neighbor_entries.items():
                    # Tentar extrair o título real do arquivo relacionado para exibição
                    neighbor_title = neighbor_file.split("/")[-1].replace(".md", "").replace("-", " ").replace("_", " ").title()
                    try:
                        n_path = Path("wiki") / neighbor_file
                        if n_path.is_file():
                            from wiki_engine import _parse_frontmatter, _extract_h1_title
                            n_text = n_path.read_text(encoding="utf-8")
                            meta, body = _parse_frontmatter(n_text)
                            title_val = meta.get("title") or _extract_h1_title(body)
                            if title_val:
                                neighbor_title = title_val
                    except Exception:
                        pass
                    
                    links_md += f"- [{neighbor_title}](../{neighbor_file})\n"

                # Limpa qualquer seção antiga de relacionados no conteúdo gerado
                content_to_write = markdown_content
                if "## Páginas Relacionadas" in content_to_write:
                    content_to_write = content_to_write.split("## Páginas Relacionadas")[0].strip()

                updated_content = content_to_write + links_md
                new_md_path.write_text(updated_content, encoding="utf-8")

                # Reindexa o arquivo atualizado com os links para manter o banco vetorial em sincronia
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: vector_db.delete_by_metadata(metadata={"name": new_md_path.name})
                )
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: knowledge.add_content(path=str(new_md_path))
                )

            # Atualiza o arquivo wiki/index.md para incluir o link do novo documento na seção correta
            def update_index_md(category: str, slug: str, title: str) -> None:
                index_path = Path("wiki/index.md")
                if not index_path.exists():
                    return
                content = index_path.read_text(encoding="utf-8")
                link_target = f"{category}/{slug}.md"
                if link_target in content:
                    return
                headers = {
                    "overviews": "## Visão Geral",
                    "entities": "## Entidades",
                    "concepts": "## Conceitos",
                    "synthesis": "## Síntese",
                    "comparisons": "## Comparações"
                }
                header = headers.get(category)
                if not header or header not in content:
                    # Fallback: anexar no final antes de '---'
                    if "---" in content:
                        parts = content.split("---")
                        parts[-2] = parts[-2].rstrip() + f"\n- [{title}]({link_target})\n\n"
                        updated = "---".join(parts)
                    else:
                        updated = content.rstrip() + f"\n\n- [{title}]({link_target})\n"
                else:
                    header_idx = content.find(header)
                    next_header_idx = len(content)
                    for h in headers.values():
                        if h == header:
                            continue
                        idx = content.find(h, header_idx + len(header))
                        if idx != -1 and idx < next_header_idx:
                            next_header_idx = idx
                    idx = content.find("---", header_idx + len(header))
                    if idx != -1 and idx < next_header_idx:
                        next_header_idx = idx
                    section_content = content[header_idx:next_header_idx]
                    lines = section_content.splitlines()
                    insert_line_idx = -1
                    for i, line in enumerate(lines):
                        if line.strip().startswith("- "):
                            insert_line_idx = i
                    if insert_line_idx != -1:
                        lines.insert(insert_line_idx + 1, f"- [{title}]({link_target})")
                        new_section = "\n".join(lines) + "\n"
                    else:
                        new_section = section_content.rstrip() + f"\n\n- [{title}]({link_target})\n\n"
                    updated = content[:header_idx] + new_section + content[next_header_idx:]
                index_path.write_text(updated, encoding="utf-8")

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: update_index_md(category, slug, title)
            )

            push("done", "✓ Concluído!", done=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            _upload_tasks[task_id]["events"].append(
                {"step": "done", "label": f"Erro: {e}", "done": True, "error": str(e)}
            )
            _upload_tasks[task_id]["finished"] = True

    asyncio.create_task(process())
    return JSONResponse({"task_id": task_id, "status": "processing"})


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
    rag_context, rag_fontes = buscar_contexto(question)

    # Combina contextos
    combined_context = ""
    all_fontes: list[str] = []
    if page_context:
        combined_context += page_context
    if rag_context:
        combined_context += f"\n\n--- Contexto adicional da base ---\n{rag_context}"
        all_fontes.extend(rag_fontes)

    conversation_history = load_conversation_history()
    history_text = build_history_text(conversation_history)

    prompt = build_travel_prompt(question, combined_context, history_text, all_fontes)

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

    return JSONResponse({"answer": answer, "page": req.page, "sources": all_fontes})


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web_app:app", host="127.0.0.1", port=port, reload=True)
