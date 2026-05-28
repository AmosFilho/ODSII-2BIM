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

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb, SearchType
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.db.sqlite import SqliteDb

# ---------- Configuração ----------
LOAD_DOCS = True              # mude para False após a primeira indexação
DOCS_PATH = "docs"            # pasta com seus PDFs
CHAT_MODEL = "llama3"      # modelo de chat do Ollama
EMBED_MODEL = "nomic-embed-text"  # modelo de embeddings

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

# ---------- Agente ----------
agent = Agent(
    name="Assistente de Documentos",
    model=Ollama(id=CHAT_MODEL),
    knowledge=knowledge,
    search_knowledge=True,
    db=SqliteDb(db_file="tmp/agent.db"),
    add_history_to_context=True,
    num_history_runs=5,
    instructions=[
        "Você é um assistente que responde com base nos documentos fornecidos.",
        "Sempre cite o trecho do documento que sustenta sua resposta.",
        "Se a informação não estiver nos documentos, diga isso claramente.",
        "Responda em português do Brasil.",
    ],
    markdown=True,
)

# ---------- Loop de chat ----------
if __name__ == "__main__":
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

        agent.print_response(pergunta, stream=True)
        print()
