"""
Gerador de páginas da wiki usando OpenClaude (OpenRouter / openai/gpt-5-mini).

Uso:
  - Defina OPENAI_API_KEY (ou outro endpoint conforme sua configuração OpenRouter).
  - Coloque fontes em raw_sources/ (txt, md, pdf).
  - Rode: python scripts/generate_wiki_openclaude.py

O script lê cada fonte, solicita ao modelo gerar uma página Markdown seguindo o esquema de CLAUDE.md
(e.g., categorias: entities, concepts, comparisons, overviews, synthesis) e salva em wiki/.
Não sobrescreve páginas existentes — cria com sufixo quando necessário.

Observação: para PDFs o script tenta usar pypdf (pypdf ou PyPDF2). Se não instalado, pula PDFs.
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    # pypdf is listed in requirements; try to import for PDF text extraction
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None

RAW_DIR = Path("raw_sources")
WIKI_DIR = Path("wiki")
INDEX_FILE = WIKI_DIR / "index.md"
MODEL = os.getenv("OPENCLAUDE_MODEL", "openai/gpt-5-mini")
API_KEY = os.getenv("OPENAI_API_KEY")

if OpenAI is None:
    print("Erro: biblioteca openai não encontrada. Instale com 'pip install openai' ou ajuste requirements.")
    sys.exit(1)

if not API_KEY:
    print("Aviso: OPENAI_API_KEY não definida. O cliente pode falhar ao chamar o modelo.")

client = OpenAI()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text)
    return text[:120]


def extract_text_from_file(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".md", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            if PdfReader is None:
                print(f"pypdf não disponível, pulando PDF: {path}")
                return None
            reader = PdfReader(str(path))
            texts = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n\n".join(texts)
    except Exception as e:
        print(f"Erro lendo {path}: {e}")
        return None
    return None


def build_prompt(source_text: str, source_name: str) -> str:
    # Instruções compactas em português (BR) seguindo o esquema CLAUDE.md
    return (
        "Você é um assistente que cria páginas de uma wiki vertical de turismo para Amazonas. "
        "Dado o conteúdo da fonte abaixo, gere UMA página Markdown pronta para ser salva em 'wiki/' seguindo estas regras:\n"
        "- Escolha a categoria mais apropriada: entities, concepts, comparisons, overviews ou synthesis.\n"
        "- Use nome de arquivo em lowercase com hífens (slug). Forneça também um título humano legível.\n"
        "- Estruture o conteúdo com seções claras: resumo, detalhes práticos (como logística, época, segurança), fontes e links cruzados sugeridos.\n"
        "- Não invente preços, horários ou disponibilidade; se houver incerteza, sinalize que precisa de verificação atual.\n"
        "- Escreva em português do Brasil.\n"
        "- No topo do arquivo inclua um pequeno frontmatter YAML opcional com: title, slug, category, source (o nome do arquivo fonte).\n\n"
        f"Fonte: {source_name}\n---BEGIN SOURCE---\n{source_text[:8000]}\n---END SOURCE---\n\n"
        "Retorne somente o conteúdo do arquivo Markdown (começando pelo frontmatter ou título) sem explicações adicionais."
    )


def generate_page_for_file(path: Path) -> Optional[Path]:
    text = extract_text_from_file(path)
    if not text:
        return None

    prompt = build_prompt(text, path.name)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Você é um assistente que escreve páginas de wiki Markdown seguindo regras editoriais."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2500,
        )
    except Exception as e:
        print(f"Erro ao chamar a API para {path.name}: {e}")
        return None

    # Extrair texto da resposta
    try:
        content = ""
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            # OpenAI-compatible response shape
            choice = resp.choices[0]
            if hasattr(choice, "message"):
                content = choice.message.get("content", "")
            elif hasattr(choice, "text"):
                content = choice.text
        else:
            # Fallback: str(resp)
            content = str(resp)
    except Exception:
        content = str(resp)

    content = content.strip()
    if not content:
        print(f"Resposta vazia do modelo para {path.name}")
        return None

    # Determine slug from frontmatter if present, else from title
    slug_match = re.search(r"slug:\s*([a-z0-9\-]+)", content, re.IGNORECASE)
    if slug_match:
        slug = slug_match.group(1).lower()
    else:
        # try to get first markdown h1
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        slug = slugify(title)

    # determine category from frontmatter
    cat_match = re.search(r"category:\s*(\w+)", content, re.IGNORECASE)
    category = cat_match.group(1).lower() if cat_match else "entities"
    if category not in {"entities", "concepts", "comparisons", "overviews", "synthesis"}:
        category = "entities"

    target_dir = WIKI_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / f"{slug}.md"
    # avoid overwrite: if exists, pick a new unique name
    if target_file.exists():
        i = 1
        while True:
            candidate = target_dir / f"{slug}_v{i}.md"
            if not candidate.exists():
                target_file = candidate
                break
            i += 1

    try:
        target_file.write_text(content, encoding="utf-8")
        print(f"Criado: {target_file}")
        return target_file
    except Exception as e:
        print(f"Erro escrevendo {target_file}: {e}")
        return None


def update_index(new_pages: list[Path]) -> None:
    if not new_pages:
        return
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if INDEX_FILE.exists():
        try:
            lines = INDEX_FILE.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []

    with INDEX_FILE.open("a", encoding="utf-8") as f:
        for p in new_pages:
            rel = p.relative_to(WIKI_DIR)
            title = p.read_text(encoding="utf-8").splitlines()[0]
            if title.startswith("---"):
                # skip frontmatter: try next line as title
                rest = p.read_text(encoding="utf-8").splitlines()
                title_line = next((l for l in rest if l.strip().startswith("#")), None)
                if title_line:
                    title = title_line
            # sanitize
            title = title.lstrip('# ').strip()
            entry = f"- [{title}]({rel.as_posix()})"
            if entry not in lines:
                f.write(entry + "\n")
                lines.append(entry)


def main() -> None:
    if not RAW_DIR.exists():
        print("Pasta raw_sources/ não encontrada. Crie-a e adicione arquivos fonte.")
        return

    files = sorted(RAW_DIR.glob("*"))
    if not files:
        print("Nenhuma fonte encontrada em raw_sources/.")
        return

    created = []
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".md", ".txt", ".pdf"}:
            print(f"Ignorando formato não suportado: {f.name}")
            continue
        page = generate_page_for_file(f)
        if page:
            created.append(page)

    update_index(created)
    print(f"Concluído. {len(created)} páginas adicionadas à wiki.")


if __name__ == "__main__":
    main()
