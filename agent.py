"""
Agente RAG local com Agno + Ollama.

Antes de rodar:
    1. Tenha o Ollama instalado e rodando (https://ollama.com)
    2. Baixe os modelos:
        ollama pull llama3.1:8b
        ollama pull nomic-embed-text
    3. Instale as dependências:
        pip install -r requirements.txt
    4. Coloque seus PDFs na pasta `docs/`
    5. Na primeira execução, deixe `LOAD_DOCS = True` para indexar.
       Depois, mude para False para não reprocessar a cada execução.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb, SearchType
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.db.sqlite import SqliteDb

try:
    from agno.tools.duckduckgo import DuckDuckGoTools
except ImportError:
    DuckDuckGoTools = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

# ---------- Configuração ----------
LOAD_DOCS = False            # mude para False após a primeira indexação
DOCS_PATH = "docs"            # pasta com seus PDFs
CHAT_MODEL = "llama3.1:8b"      # modelo de chat do Ollama
EMBED_MODEL = "nomic-embed-text:v1.5"  # modelo de embeddings
DB_FILE = "tmp/agent.db"
SESSION_ID = "assistente_documentos"
USER_ID = "usuario_local"
HISTORY_MESSAGES = 20
HISTORY_FILE = "tmp/conversation_history.json"
MAX_HISTORY_ENTRIES = 10
AUDIO_OUTPUT_DIR = "tmp/audio"
OPENAI_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
OPENAI_VOICE_RESPONSE_MODEL = os.getenv("OPENAI_VOICE_RESPONSE_MODEL", "gpt-4o-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

os.makedirs("tmp", exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# ---------- Vector store local (ChromaDB) ----------
vector_db = ChromaDb(
    collection="documentos",
    path="tmp/chromadb",
    persistent_client=True,
    search_type=SearchType.hybrid,
    embedder=OllamaEmbedder(
        id=EMBED_MODEL,
        dimensions=768,
    ),
)

# ---------- Base de conhecimento ----------
knowledge = Knowledge(
    name="Base de documentos local",
    description="Documentos PDF carregados localmente",
    vector_db=vector_db,
)

if LOAD_DOCS:
    print(f"Indexando documentos da pasta '{DOCS_PATH}/' ...")
    knowledge.add_content(path=DOCS_PATH)
    print("Indexação concluída.\n")

def responder_comando_de_voz(
    audio_path: str,
    output_path: Optional[str] = None,
    voice: str = OPENAI_TTS_VOICE,
) -> str:
    """
    Recebe um arquivo de audio com um comando de voz, transcreve o comando,
    gera uma resposta curta com IA e salva a resposta em audio.
    """

    if OpenAI is None:
        return "A biblioteca openai nao esta instalada. Rode: pip install -r requirements.txt"

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY nao configurada. Defina essa variavel de ambiente para usar voz."

    audio_file_path = Path(audio_path).expanduser()
    if not audio_file_path.exists():
        return f"Arquivo de audio nao encontrado: {audio_file_path}"

    client = OpenAI()

    with audio_file_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIPTION_MODEL,
            file=audio_file,
            response_format="text",
        )

    comando = str(transcript).strip()
    if not comando:
        return "Nao consegui transcrever nenhum comando de voz nesse audio."

    response = client.chat.completions.create(
        model=OPENAI_VOICE_RESPONSE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Responda em portugues do Brasil, de forma direta e natural.",
            },
            {"role": "user", "content": comando},
        ],
    )
    resposta_texto = response.choices[0].message.content or ""
    resposta_texto = resposta_texto.strip()

    if output_path:
        audio_output_path = Path(output_path).expanduser()
    else:
        audio_count = len(list(Path(AUDIO_OUTPUT_DIR).glob("*.mp3"))) + 1
        audio_output_path = Path(AUDIO_OUTPUT_DIR) / f"resposta_voz_{audio_count}.mp3"

    audio_output_path.parent.mkdir(parents=True, exist_ok=True)
    speech = client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=voice,
        input=resposta_texto,
        response_format="mp3",
    )
    audio_output_path.write_bytes(speech.content)

    return json.dumps(
        {
            "comando_transcrito": comando,
            "resposta_texto": resposta_texto,
            "audio_resposta": str(audio_output_path),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_tools() -> List[Any]:
    tools: List[Any] = [responder_comando_de_voz]

    if DuckDuckGoTools is not None:
        tools.append(
            DuckDuckGoTools(
                enable_search=True,
                enable_news=True,
                fixed_max_results=5,
                region="br-pt",
            )
        )

    return tools


# ---------- Agente ----------
agent = Agent(
    name="Assistente de Documentos",
    model=Ollama(id=CHAT_MODEL),
    knowledge=knowledge,
    tools=build_tools(),
    search_knowledge=False,
    db=SqliteDb(db_file=DB_FILE),
    session_id=SESSION_ID,
    user_id=USER_ID,
    add_history_to_context=True,
    store_history_messages=True,
    add_session_state_to_context=True,
    overwrite_db_session_state=False,
    num_history_runs=5,
    instructions=[
        "Você é um assistente que responde com base nos documentos fornecidos.",
        "Sempre cite o trecho do documento que sustenta sua resposta.",
        "Se a informação não estiver nos documentos, diga isso claramente.",
        "Responda em português do Brasil.",
        "Use a tool de web search quando o usuario pedir informacoes atuais ou fora dos documentos.",
        "Use a tool responder_comando_de_voz quando o usuario informar um caminho de arquivo de audio e pedir resposta em voz.",
    ],
    debug_mode=True,
    markdown=True,
)

session = agent.get_session(session_id=SESSION_ID, user_id=USER_ID)
if session:
    print(f"Histórico persistente carregado para a sessão '{SESSION_ID}'.")
else:
    print(f"Criando nova sessão persistente '{SESSION_ID}'.")


def load_conversation_history() -> List[Dict[str, str]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_conversation_history(history: List[Dict[str, str]]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ---------- Função de RAG manual ----------
def buscar_contexto(pergunta: str, top_k: int = 3) -> str:
    """
    Busca os documentos relevantes no banco vetorial
    e retorna apenas o conteúdo textual.
    """

    resultados = vector_db.search(pergunta, limit=top_k)

    contextos = []

    for i, doc in enumerate(resultados, start=1):

        conteudo = ""

        # tenta pegar conteúdo de forma segura
        if hasattr(doc, "content"):
            conteudo = doc.content

        elif isinstance(doc, dict):
            conteudo = doc.get("content", "")

        if conteudo:
            contextos.append(
                f"[Trecho {i}]\n{conteudo}"
            )

    return "\n\n".join(contextos)


def build_history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""

    lines = []
    for item in history[-MAX_HISTORY_ENTRIES:]:
        role = "Usuário" if item["role"] == "user" else "Assistente"
        lines.append(f"{role}: {item['content']}")

    return "\n".join(lines)


# ---------- Loop de chat ----------
if __name__ == "__main__":
    conversation_history = load_conversation_history()

    print("Assistente pronto. Digite sua pergunta (ou 'sair' para encerrar).\n")
    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break
        if pergunta.lower().startswith("/voz "):
            audio_path = pergunta[5:].strip().strip('"')
            print("\nAssistente:\n")
            print(responder_comando_de_voz(audio_path))
            print("\n")
            continue

        # ---------- Retrieval manual ----------
        contexto = buscar_contexto(pergunta)
        history_text = build_history_text(conversation_history)
        history_section = (
            f"Histórico da conversa anterior:\n{history_text}\n\n" if history_text else ""
        )

        # ---------- Prompt manual ----------
        prompt = f"""
Você é um assistente acadêmico especializado em responder perguntas
com base em documentos científicos.

{history_section}Use APENAS o contexto abaixo para responder.

================ CONTEXTO ================

{contexto}

==========================================

Pergunta:
{pergunta}

REGRAS IMPORTANTES:
- Responda diretamente à pergunta
- NÃO descreva os documentos
- NÃO explique metadados
- NÃO diga "o conjunto de dados parece..."
- NÃO descreva estrutura JSON
- NÃO invente informações
- Se a resposta não estiver no contexto, diga isso claramente
- Responda em português do Brasil
- Cite os autores e trabalhos quando possível
"""

        print("\nAssistente:\n")

        response = agent.run(
            prompt,
            session_id=SESSION_ID,
            add_history_to_context=True,
            add_session_state_to_context=True,
            stream=False,
        )

        answer = response.content if response.content is not None else ""
        print(answer)
        print("\n")

        conversation_history.append({"role": "user", "content": pergunta})
        conversation_history.append({"role": "assistant", "content": answer})
        save_conversation_history(conversation_history)
