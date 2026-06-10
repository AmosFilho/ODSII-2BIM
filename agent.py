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
os.makedirs("tmp", exist_ok=True)

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


def build_tools() -> List[Any]:
    return []


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
        "Nao invente precos, horarios, disponibilidade de passeios, condicoes de estrada, regras sanitarias ou exigencias legais.",
        "Responda em portugues do Brasil, com tom pratico e acolhedor.",
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


def safe_agent_run(prompt: str, session_id: str = SESSION_ID) -> str:
    """
    Executa o agente de forma direta.
    """
    result = agent.run(
        prompt,
        session_id=session_id,
        add_history_to_context=False,
        add_session_state_to_context=False,
        stream=False,
    )
    return result.content if result.content is not None else ""


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
