# wiki_graph.py
import json
from pathlib import Path

GRAPH_PATH = Path(".obsidian/graph.json")

def load_graph() -> dict:
    if not GRAPH_PATH.exists():
        return {}
    try:
        return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_graph(graph: dict) -> None:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")

def update_page_relations(page_path: str, neighbors_with_scores: dict[str, float]) -> None:
    """
    Atualiza as conexões de uma página no grafo de forma incremental e bidirecional.
    Salva as conexões sob a chave 'links' em .obsidian/graph.json.
    """
    graph = load_graph()
    if "links" not in graph:
        graph["links"] = {}
    
    links = graph["links"]
    old_neighbors = set(links.get(page_path, {}).keys())
    new_neighbors = set(neighbors_with_scores.keys())
    
    # Atualiza as conexões da página atual
    if neighbors_with_scores:
        links[page_path] = neighbors_with_scores
    else:
        links.pop(page_path, None)
        
    # Adiciona a nova página aos vizinhos (bidirecionalidade)
    for neighbor in new_neighbors:
        if neighbor not in links:
            links[neighbor] = {}
        links[neighbor][page_path] = neighbors_with_scores[neighbor]
        
    # Remove a página das conexões que foram desfeitas
    for old_neighbor in (old_neighbors - new_neighbors):
        if old_neighbor in links:
            links[old_neighbor].pop(page_path, None)
            if not links[old_neighbor]:
                links.pop(old_neighbor, None)
                
    save_graph(graph)

def get_page_relations(page_path: str) -> dict[str, float]:
    """
    Retorna um dicionário {neighbor_path: similarity_score} para a página informada.
    """
    graph = load_graph()
    return graph.get("links", {}).get(page_path, {})
