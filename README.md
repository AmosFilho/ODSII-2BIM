# AmazonIA Travel

IA vertical com RAG para turismo e viagens no Amazonas. O projeto combina uma wiki local em Markdown, busca vetorial com ChromaDB/embeddings via Ollama e uma interface web em FastAPI para responder perguntas sobre destinos, roteiros, sazonalidade, logistica, seguranca e boas praticas.

## Como rodar

1. Instale e inicie o Ollama.
2. Baixe os modelos:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

3. Instale as dependencias:

```powershell
pip install -r requirements.txt
```

4. Para indexar a wiki pela primeira vez, altere `LOAD_DOCS = True` em `agent.py` e execute:

```powershell
python agent.py
```

5. Depois da indexacao, volte `LOAD_DOCS = False` e rode a interface:

```powershell
python web_app.py
```

Acesse `http://127.0.0.1:8000`.

## Base de conhecimento

A base principal fica em `wiki/`:

- `wiki/index.md`: indice da wiki.
- `wiki/overviews/`: visoes gerais do dominio.
- `wiki/entities/`: destinos e experiencias.
- `wiki/concepts/`: conceitos de planejamento.
- `wiki/synthesis/`: roteiros e respostas sinteticas.

Informacoes dinamicas, como precos, horarios, disponibilidade, eventos e regras recentes, devem ser verificadas em fontes atuais.
