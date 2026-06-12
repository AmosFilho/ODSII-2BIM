"""
Wiki Engine — funções de navegação, renderização e consulta da wiki.

Fornece:
- Escaneamento da árvore de páginas (wiki/)
- Conversão Markdown → HTML com sumário automático
- Busca por título/conteúdo
- Descoberta de páginas relacionadas por overlap de palavras-chave
- Construção do índice JSON para navegação no frontend
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension

import wiki_graph

WIKI_DIR = Path("wiki")

# ── Frontmatter parsing ──────────────────────────────────────────────────────

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_WORD_RE = re.compile(r"\b[a-zA-ZÀ-ÿ]{4,}\b")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extrai YAML frontmatter simples (chave: valor) e retorna (meta, body)."""
    match = _FM_RE.match(text)
    if not match:
        return {}, text
    raw = match.group(1)
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    body = text[match.end():]
    return meta, body


# ── Markdown → HTML ──────────────────────────────────────────────────────────

_md = markdown.Markdown(
    extensions=[
        TocExtension(permalink=False),
        FencedCodeExtension(),
        CodeHiliteExtension(css_class="highlight", noclasses=True),
        "tables",
        "attr_list",
    ]
)


def render_markdown_to_html(text: str) -> tuple[str, str]:
    """Converte Markdown para HTML, removendo links de imagens e links externos.
    Mantém links internos (relativos) e formata o texto limpo.
    """
    # Remove sintaxe de imagens Markdown: ![alt](url)
    text = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', text)
    # Remove links externos, mantendo apenas o texto âncora
    def _strip_external(match: re.Match) -> str:
        anchor, url = match.group(1), match.group(2)
        if url.startswith('http://') or url.startswith('https://'):
            return anchor  # devolve só o texto da âncora
        return match.group(0)  # mantém o link original
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _strip_external, text)
    _md.reset()
    html = _md.convert(text)
    return html, _md.toc
    """
    Converte Markdown para HTML.
    Retorna (html, toc_html) — o sumário gerado pela extensão TOC.
    """
    _md.reset()
    html = _md.convert(text)
    return html, _md.toc


# ── Chunking específico para Markdown ────────────────────────────────────────

def chunk_markdown(text: str, max_chars: int = 1200) -> list[str]:
    """
    Divide um documento Markdown em chunks respeitando a estrutura de headers.

    Estratégia:
    1. Divide nos headers # / ## / ### mantendo o texto de cada seção junto.
    2. Seções maiores que max_chars são subdivididas por parágrafos duplos.
    3. Retorna lista de strings prontas para embedding.
    """
    # Padrão: qualquer linha que começa com # (H1–H6)
    header_pattern = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)

    # Encontra as posições de cada header
    splits = [m.start() for m in header_pattern.finditer(text)]

    if not splits:
        # Sem headers: usa divisão por parágrafo
        return _split_by_paragraphs(text, max_chars)

    # Garante que capturamos o texto antes do 1º header (introdução)
    sections: list[str] = []
    if splits[0] > 0:
        intro = text[: splits[0]].strip()
        if intro:
            sections.append(intro)

    for idx, start in enumerate(splits):
        end = splits[idx + 1] if idx + 1 < len(splits) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)

    # Subdivide seções longas por parágrafo
    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            chunks.extend(_split_by_paragraphs(section, max_chars))

    return [c for c in chunks if c.strip()]


def _split_by_paragraphs(text: str, max_chars: int) -> list[str]:
    """Divide texto por parágrafos duplos; agrega fragmentos pequenos."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Parágrafo isolado maior que max_chars: mantém como chunk único
            current = para
    if current:
        chunks.append(current)
    return chunks


# ── Sumário de headers ───────────────────────────────────────────────────────

def extract_page_summary(text: str) -> list[dict[str, Any]]:
    """
    Extrai headers do Markdown como sumário.
    Retorna lista de {"level": int, "title": str, "anchor": str}.
    """
    summary: list[dict[str, Any]] = []
    for match in _HEADER_RE.finditer(text):
        level = len(match.group(1))
        title = match.group(2).strip()
        # Gera anchor compatível com a extensão TOC do Python-Markdown
        anchor = re.sub(r"[^a-zA-ZÀ-ÿ0-9 _-]", "", title)
        anchor = anchor.lower().replace(" ", "-")
        summary.append({"level": level, "title": title, "anchor": anchor})
    return summary


# ── Escaneamento de páginas ──────────────────────────────────────────────────

def _slug_to_title(slug: str) -> str:
    """Converte 'amazonas-destinations' → 'Amazonas Destinations'."""
    return slug.replace("-", " ").replace("_", " ").title()


def _extract_h1_title(body: str) -> str | None:
    """Extrai o primeiro heading H1 do corpo markdown."""
    match = re.search(r"^#\s+(.+)$", body.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def list_wiki_pages() -> dict[str, list[dict[str, Any]]]:
    """
    Escaneia wiki/ e retorna árvore de categorias → páginas.
    Ignora index.md no nível raiz.
    """
    tree: dict[str, list[dict[str, Any]]] = {}
    if not WIKI_DIR.is_dir():
        return tree

    for category_dir in sorted(WIKI_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        cat_name = category_dir.name
        pages: list[dict[str, Any]] = []
        for md_file in sorted(category_dir.glob("*.md")):
            slug = md_file.stem
            raw = md_file.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(raw)
            title = meta.get("title") or _extract_h1_title(body) or _slug_to_title(slug)
            summary = extract_page_summary(body)
            pages.append(
                {
                    "slug": slug,
                    "title": title,
                    "file": f"{cat_name}/{slug}.md",
                    "summary": summary,
                    "meta": meta,
                }
            )
        if pages:
            tree[cat_name] = pages

    return tree


def get_page_content(category: str, slug: str) -> dict[str, Any] | None:
    """
    Lê uma página da wiki e retorna dicionário com conteúdo.
    """
    md_path = WIKI_DIR / category / f"{slug}.md"
    if not md_path.is_file():
        return None

    raw = md_path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    html, toc = render_markdown_to_html(body)
    title = meta.get("title") or _extract_h1_title(body) or _slug_to_title(slug)
    summary = extract_page_summary(body)

    return {
        "slug": slug,
        "title": title,
        "category": category,
        "html": html,
        "toc": toc,
        "toc_items": summary,
        "markdown": body,
        "meta": meta,
    }


# ── Busca ────────────────────────────────────────────────────────────────────

def search_wiki(query: str) -> list[dict[str, Any]]:
    """
    Busca páginas da wiki por título ou conteúdo (case-insensitive).
    Retorna lista de resultados com snippet.
    """
    query_lower = query.lower()
    results: list[dict[str, Any]] = []
    tree = list_wiki_pages()

    for cat_name, pages in tree.items():
        for page in pages:
            md_path = WIKI_DIR / cat_name / f"{page['slug']}.md"
            raw = md_path.read_text(encoding="utf-8")
            _, body = _parse_frontmatter(raw)
            body_lower = body.lower()

            score = 0
            # Match no título tem peso maior
            if query_lower in page["title"].lower():
                score += 10
            # Match no slug
            if query_lower in page["slug"].lower():
                score += 5
            # Match no corpo
            count = body_lower.count(query_lower)
            score += count

            if score > 0:
                # Extrai snippet ao redor da primeira ocorrência
                idx = body_lower.find(query_lower)
                start = max(0, idx - 60)
                end = min(len(body), idx + len(query) + 60)
                snippet = ("..." if start > 0 else "") + body[start:end] + ("..." if end < len(body) else "")
                results.append(
                    {
                        "slug": page["slug"],
                        "title": page["title"],
                        "category": cat_name,
                        "file": f"{cat_name}/{page['slug']}.md",
                        "snippet": snippet.strip(),
                        "score": score,
                    }
                )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ── Páginas relacionadas ─────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Extrai palavras significativas (4+ chars) em lowercase."""
    return {m.lower() for m in _WORD_RE.findall(text)}


def _jaccard_related(category: str, slug: str, max_results: int) -> list[dict[str, Any]]:
    """Fallback: Jaccard simples sobre tokens quando a página não está no grafo."""
    target_path = WIKI_DIR / category / f"{slug}.md"
    if not target_path.is_file():
        return []

    target_raw = target_path.read_text(encoding="utf-8")
    _, target_body = _parse_frontmatter(target_raw)
    target_tokens = _tokenize(target_body)

    if not target_tokens:
        return []

    tree = list_wiki_pages()
    scored: list[dict[str, Any]] = []

    for cat_name, pages in tree.items():
        for page in pages:
            if cat_name == category and page["slug"] == slug:
                continue
            other_path = WIKI_DIR / cat_name / f"{page['slug']}.md"
            other_raw = other_path.read_text(encoding="utf-8")
            _, other_body = _parse_frontmatter(other_raw)
            other_tokens = _tokenize(other_body)

            if not other_tokens:
                continue

            intersection = target_tokens & other_tokens
            union = target_tokens | other_tokens
            jaccard = len(intersection) / len(union) if union else 0

            if jaccard > 0:
                scored.append(
                    {
                        "slug": page["slug"],
                        "title": page["title"],
                        "category": cat_name,
                        "file": f"{cat_name}/{page['slug']}.md",
                        "score": round(jaccard, 4),
                    }
                )

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:max_results]


def find_related_pages(category: str, slug: str, max_results: int = 3) -> list[dict[str, Any]]:
    """
    Encontra páginas relacionadas consultando o grafo de relacionamentos (.obsidian/graph.json).
    Se a página não estiver no grafo, usa Jaccard como fallback.
    """
    page_path = f"{category}/{slug}.md"
    relations = wiki_graph.get_page_relations(page_path)

    if relations:
        tree = list_wiki_pages()
        all_pages = {}
        for cat_name, pages in tree.items():
            for p in pages:
                all_pages[p["file"]] = p

        scored = []
        for file_path, score in relations.items():
            if file_path in all_pages:
                page_info = all_pages[file_path]
                scored.append({
                    "slug": page_info["slug"],
                    "title": page_info["title"],
                    "category": file_path.split("/")[0],
                    "file": file_path,
                    "score": round(score, 4),
                })
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:max_results]

    # Fallback para Jaccard
    return _jaccard_related(category, slug, max_results)



# ── Índice JSON ──────────────────────────────────────────────────────────────

def build_wiki_index_data() -> dict[str, Any]:
    """
    Constrói JSON completo do índice da wiki para navegação no frontend.
    Combina list_wiki_pages() com metadados do index.md.
    """
    tree = list_wiki_pages()

    # Lê index.md se existir
    index_md = ""
    index_path = WIKI_DIR / "index.md"
    if index_path.is_file():
        raw = index_path.read_text(encoding="utf-8")
        _, index_md = _parse_frontmatter(raw)

    index_html = ""
    if index_md:
        index_html, _ = render_markdown_to_html(index_md)

    total_pages = sum(len(pages) for pages in tree.values())

    return {
        "categories": [
            {
                "name": cat,
                "label": cat.replace("-", " ").title(),
                "pages": pages,
            }
            for cat, pages in tree.items()
        ],
        "index_html": index_html,
        "total_pages": total_pages,
    }
