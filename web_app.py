from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from agent import agent, SESSION_ID
from agent import load_conversation_history, save_conversation_history, build_history_text, buscar_contexto

app = FastAPI(title="Assistente de Documentos")
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
  <title>Assistente de Documentos</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f4f9; color: #111; margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 0 auto; padding: 24px; }
    h1 { margin-bottom: 8px; }
    .chat-window { background: #fff; border-radius: 12px; border: 1px solid #dcdce1; padding: 16px; min-height: 400px; box-shadow: 0 4px 10px rgba(0,0,0,0.04); overflow-y: auto; }
    .message { margin-bottom: 16px; }
    .message.user { text-align: right; }
    .message .bubble { display: inline-block; padding: 12px 16px; border-radius: 16px; max-width: 85%; white-space: pre-wrap; }
    .message.user .bubble { background: #2f80ed; color: #fff; border-bottom-right-radius: 4px; }
    .message.assistant .bubble { background: #e0e7ff; color: #111; border-bottom-left-radius: 4px; }
    .controls { display: flex; gap: 8px; margin-top: 16px; }
    input[type="text"] { flex: 1; padding: 12px 14px; border-radius: 999px; border: 1px solid #ccd0db; font-size: 16px; }
    button { border: none; border-radius: 999px; background: #2f80ed; color: #fff; padding: 12px 18px; cursor: pointer; font-size: 16px; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .footer { margin-top: 14px; font-size: 14px; color: #555; }
    .status { margin-bottom: 12px; color: #333; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Assistente de Documentos</h1>
    <div class="status">Use o chat abaixo ou clique no ícone de microfone para ditar sua pergunta.</div>
    <div id="chat" class="chat-window"></div>

    <div class="controls">
      <input id="message" type="text" placeholder="Digite sua pergunta aqui..." autocomplete="off" />
      <button id="voiceBtn">🎙️ Voz</button>
      <button id="sendBtn">Enviar</button>
    </div>
    <div class="footer">A interface usa o reconhecimento de voz do navegador (Web Speech API).</div>
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
        addMessage('assistant', data.answer || 'Não foi possível obter resposta.');
      } catch (error) {
        addMessage('assistant', 'Erro ao chamar o servidor: ' + error.message);
      } finally {
        sendBtn.disabled = false;
        voiceBtn.disabled = false;
      }
    };

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
        voiceBtn.textContent = '🎙️ Voz';
        voiceBtn.disabled = false;
      });
    } else {
      voiceBtn.disabled = true;
      voiceBtn.textContent = 'Voz não suportada';
    }

    voiceBtn.addEventListener('click', () => {
      if (!recognition) return;
      voiceBtn.disabled = true;
      voiceBtn.textContent = 'Ouvindo...';
      recognition.start();
    });
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
        return JSONResponse({"answer": "Por favor, envie uma pergunta válida."}, status_code=400)

    conversation_history = load_conversation_history()
    history_text = build_history_text(conversation_history)
    history_section = f"Histórico da conversa anterior:\n{history_text}\n\n" if history_text else ""
    contexto = buscar_contexto(message)

    prompt = f"""
Você é um assistente acadêmico especializado em responder perguntas
com base em documentos científicos.

{history_section}Use APENAS o contexto abaixo para responder.

================ CONTEXTO ================

{contexto}

==========================================

Pergunta:
{message}

REGRAS IMPORTANTES:
- Responda diretamente à pergunta
- NÃO descreva os documentos
- NÃO explique metadados
- NÃO diga \"o conjunto de dados parece...\"
- NÃO descreva estrutura JSON
- NÃO invente informações
- Se a resposta não estiver no contexto, diga isso claramente
- Responda em português do Brasil
- Cite os autores e trabalhos quando possível
"""

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
