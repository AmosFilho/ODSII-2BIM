from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import shutil

from agent import SESSION_ID, agent, knowledge
from agent import build_history_text, build_travel_prompt, buscar_contexto
from agent import load_conversation_history, save_conversation_history

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

    .shell {
      min-height: 100vh;
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

    @media (max-width: 820px) {
      .shell { grid-template-columns: 1fr; }
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
  <div class="shell">
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

  <script>
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


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("web_app:app", host="127.0.0.1", port=port, reload=True)
