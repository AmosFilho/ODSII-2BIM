"""
Agente RAG vertical para turismo e viagens no Amazonas com Agno + Ollama.

Antes de rodar:
    1. Tenha o Ollama instalado e rodando (https://ollama.com)
    2. Baixe os modelos:
        ollama pull llama3.1:8b
        ollama pull nomic-embed-text
    3. Instale as dependencias:
        pip install -r requirements.txt
    4. Mantenha a base de conhecimento em `wiki/`.
    5. Na primeira execucao, deixe `LOAD_DOCS = True` para indexar.
       Depois, mude para False para nao reprocessar a cada execucao.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.vectordb.chroma import ChromaDb, SearchType

try:
    from agno.tools.duckduckgo import DuckDuckGoTools
except ImportError:
    DuckDuckGoTools = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

# ---------- Configuracao ----------
LOAD_DOCS = False
DOCS_PATH = "wiki"
CHAT_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text:v1.5"
DB_FILE = "tmp/agent.db"
SESSION_ID = "amazonas_travel_rag"
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
    collection="turismo_amazonas",
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
    name="Wiki AmazonIA Travel",
    description="Base vertical de turismo, viagem e planejamento no Amazonas",
    vector_db=vector_db,
)

if LOAD_DOCS:
    print(f"Indexando base de conhecimento da pasta '{DOCS_PATH}/' ...")
    knowledge.add_content(path=DOCS_PATH)
    print("Indexacao concluida.\n")


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
                "content": (
                    "Responda em portugues do Brasil, como uma IA de turismo "
                    "especializada no Amazonas. Seja direta, natural e segura."
                ),
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
    name="AmazonIA Travel",
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
        "Voce e a AmazonIA Travel, uma IA vertical especializada em turismo e viagens para o Amazonas.",
        "Use a wiki local como fonte principal para destinos, roteiros, clima, logistica, cultura, seguranca e boas praticas.",
        "Raciocine como consultor de viagem: considere perfil do viajante, tempo disponivel, orcamento, epoca do ano e restricoes.",
        "Sempre cite o trecho ou pagina da base que sustenta sua resposta quando houver contexto recuperado.",
        "Se a informacao nao estiver na base, diga isso claramente e sinalize quando uma verificacao atual for necessaria.",
        "Use web search quando o usuario pedir informacoes atuais, precos, horarios, disponibilidade, regras recentes, eventos ou noticias.",
        "Nao invente precos, horarios, disponibilidade de passeios, condicoes de estrada, regras sanitarias ou exigencias legais.",
        "Responda em portugues do Brasil, com tom pratico e acolhedor.",
        "Use a tool responder_comando_de_voz quando o usuario informar um caminho de arquivo de audio e pedir resposta em voz.",
    ],
    debug_mode=True,
    markdown=True,
)

session = agent.get_session(session_id=SESSION_ID, user_id=USER_ID)
if session:
    print(f"Historico persistente carregado para a sessao '{SESSION_ID}'.")
else:
    print(f"Criando nova sessao persistente '{SESSION_ID}'.")


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


# ---------- Funcao de RAG manual ----------
def buscar_contexto(pergunta: str, top_k: int = 4) -> str:
    """
    Busca os trechos relevantes na wiki vetorial
    e retorna apenas o conteudo textual.
    """

    resultados = vector_db.search(pergunta, limit=top_k)
    contextos = []

    for i, doc in enumerate(resultados, start=1):
        conteudo = ""

        if hasattr(doc, "content"):
            conteudo = doc.content
        elif isinstance(doc, dict):
            conteudo = doc.get("content", "")

        if conteudo:
            contextos.append(f"[Trecho {i}]\n{conteudo}")

    return "\n\n".join(contextos)


def build_history_text(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""

    lines = []
    for item in history[-MAX_HISTORY_ENTRIES:]:
        role = "Usuario" if item["role"] == "user" else "Assistente"
        lines.append(f"{role}: {item['content']}")

    return "\n".join(lines)


def build_travel_prompt(pergunta: str, contexto: str, history_text: str = "") -> str:
    history_section = (
        f"Historico da conversa anterior:\n{history_text}\n\n" if history_text else ""
    )

    return f"""
Voce e a AmazonIA Travel, uma IA vertical especializada em turismo e viagens
para o Amazonas. Sua tarefa e ajudar viajantes, guias, agencias e gestores
a planejar experiencias responsaveis no estado.

{history_section}Use o contexto abaixo como fonte principal para responder.

================ CONTEXTO ================

{contexto}

==========================================

Pergunta do viajante:
{pergunta}

REGRAS IMPORTANTES:
- Responda diretamente a pergunta, com orientacao pratica
- Quando fizer sentido, organize por roteiro, epoca, deslocamento, custo relativo, riscos e proximos passos
- Nao descreva metadados nem estrutura interna dos documentos
- Nao invente informacoes
- Se a resposta nao estiver no contexto, diga isso claramente e recomende verificacao atual
- Para precos, horarios, disponibilidade, regras recentes e eventos, indique que e preciso consultar fonte atual
- Responda em portugues do Brasil
- Cite paginas ou trechos da base quando possivel
"""


# ---------- Fallback para tool calls não processados ----------
def _is_raw_tool_call(text: str) -> bool:
    """
    Detecta se o texto parece ser um JSON de tool call não processado
    (quando o modelo retorna a tool call como texto em vez de executá-la).
    """
    if not text:
        return False
    stripped = text.strip()
    # Remove code blocks se presentes
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            name = data.get("name", "")
            params = data.get("parameters", data.get("arguments", {}))
            if name and isinstance(params, dict):
                return True
    except (json.JSONDecodeError, ValueError):
        pass
    return False


def _extract_search_query_from_tool_call(text: str) -> str:
    """Extrai a query de busca de um JSON de tool call."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        data = json.loads(stripped)
        params = data.get("parameters", data.get("arguments", {}))
        if isinstance(params, dict):
            return params.get("query", params.get("search", ""))
    except (json.JSONDecodeError, ValueError):
        pass
    return ""


def safe_agent_run(prompt: str, session_id: str = SESSION_ID) -> str:
    """
    Executa o agente e, se o modelo retornar uma tool call não processada
    (JSON bruto em vez de resultado), faz fallback: executa a web search
    diretamente e reenvia ao agente para gerar a resposta final.
    """
    result = agent.run(
        prompt,
        session_id=session_id,
        add_history_to_context=False,
        add_session_state_to_context=False,
        stream=False,
    )

    answer = result.content if result.content is not None else ""

    # Se a resposta é uma tool call não processada, faz fallback
    if _is_raw_tool_call(answer):
        search_query = _extract_search_query_from_tool_call(answer)
        if search_query and DuckDuckGoTools is not None:
            try:
                ddg = DuckDuckGoTools()
                search_results = ddg.web_search(query=search_query, max_results=5)
                if search_results:
                    web_context = f"\n\n--- Resultados da busca por '{search_query}' ---\n{search_results}"
                    retry_prompt = (
                        f"{prompt}\n\n{web_context}\n\n"
                        "Com base nos resultados da busca acima, responda a pergunta do viajante."
                    )
                    retry_result = agent.run(
                        retry_prompt,
                        session_id=session_id,
                        add_history_to_context=False,
                        add_session_state_to_context=False,
                        stream=False,
                    )
                    answer = retry_result.content if retry_result.content is not None else answer
            except Exception:
                answer = (
                    "Desculpe, não consegui buscar informações atualizadas no momento. "
                    "Por favor, tente novamente em instantes."
                )

    return answer


# ---------- Loop de chat ----------
if __name__ == "__main__":
    conversation_history = load_conversation_history()

    print("AmazonIA Travel pronta. Digite sua pergunta (ou 'sair' para encerrar).\n")
    while True:
        try:
            pergunta = input("Voce: ").strip()
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
            print("\nAmazonIA Travel:\n")
            print(responder_comando_de_voz(audio_path))
            print("\n")
            continue

        contexto = buscar_contexto(pergunta)
        history_text = build_history_text(conversation_history)
        prompt = build_travel_prompt(pergunta, contexto, history_text)

        print("\nAmazonIA Travel:\n")

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
