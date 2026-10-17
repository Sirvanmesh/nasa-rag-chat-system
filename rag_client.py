"""Retrieval client: connects to ChromaDB, searches by question embedding, and builds LLM context."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from llm_client import get_openai_client

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHARS_PER_DOCUMENT = 1500

MISSION_LABELS = {
    "apollo_11": "Apollo 11",
    "apollo_13": "Apollo 13",
    "challenger": "Challenger (STS-51L)",
}

PROJECT_DIR = Path(__file__).resolve().parent


def get_chroma_client(chroma_dir: str) -> chromadb.PersistentClient:
    """Open a persistent ChromaDB client with telemetry turned off."""
    return chromadb.PersistentClient(path=str(chroma_dir),
                                     settings=Settings(anonymized_telemetry=False))


def discover_chroma_backends(base_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(base_dir) if base_dir else PROJECT_DIR

    chroma_dirs = [path for path in current_dir.iterdir()
                   if path.is_dir() and "chroma" in path.name.lower()]

    for chroma_dir in chroma_dirs:
        try:
            client = get_chroma_client(chroma_dir)
            for collection in client.list_collections():
                try:
                    count = collection.count()
                except Exception:
                    count = "unknown"
                key = f"{chroma_dir.name}::{collection.name}"
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{collection.name} ({chroma_dir.name}, {count} chunks)",
                    "document_count": count,
                }
        except Exception as error:
            backends[chroma_dir.name] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {str(error)[:50]}...)",
                "document_count": 0,
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[Optional[object], bool, str]:
    """Open the ChromaDB collection. Returns (collection, success, error_message)."""
    try:
        client = get_chroma_client(chroma_dir)
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as error:
        return None, False, str(error)


def embed_query(query: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """Embed the question with the same OpenAI model that was used for the chunks."""
    response = get_openai_client().embeddings.create(model=model, input=query)
    return response.data[0].embedding


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    if not query or not query.strip():
        raise ValueError("Query is empty.")
    if n_results < 1:
        raise ValueError("n_results must be at least 1.")

    where_filter = None
    if mission_filter and mission_filter.lower() not in ("all", "all missions"):
        where_filter = {"mission": mission_filter}

    embedding_model = (collection.metadata or {}).get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    query_embedding = embed_query(query, embedding_model)

    # Ask for extra candidates so we still have n_results after removing duplicates.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 2,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    return deduplicate_results(results, n_results)


def deduplicate_results(results: Dict, n_results: int) -> Dict:
    """Sort results by distance (closest first), drop repeated texts, and keep the top n."""
    rows = sorted(
        zip(results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]),
        key=lambda row: row[3],
    )

    seen_texts = set()
    unique_rows = []
    for row in rows:
        text_key = row[1].strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        unique_rows.append(row)
        if len(unique_rows) == n_results:
            break

    return {
        "ids": [[row[0] for row in unique_rows]],
        "documents": [[row[1] for row in unique_rows]],
        "metadatas": [[row[2] for row in unique_rows]],
        "distances": [[row[3] for row in unique_rows]],
    }


def format_mission_name(mission: str) -> str:
    """Turn a stored mission id like 'apollo_13' into 'Apollo 13'."""
    return MISSION_LABELS.get(mission, mission.replace("_", " ").title())


def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    rows = list(zip(documents, metadatas, distances or [None] * len(documents)))
    if distances:
        rows.sort(key=lambda row: row[2])

    context_parts = ["=== Retrieved NASA mission documents ==="]
    seen_texts = set()
    source_number = 0

    for document, metadata, distance in rows:
        if document.strip() in seen_texts:
            continue
        seen_texts.add(document.strip())
        source_number += 1

        mission = format_mission_name(metadata.get("mission", "unknown"))
        source = metadata.get("source", "unknown source")
        category = metadata.get("document_category", "general").replace("_", " ").title()

        header = f"[Source {source_number}] Mission: {mission} | Source: {source} | Category: {category}"
        if distance is not None:
            header += f" | Relevance: {1 - distance:.2f}"
        context_parts.append(header)

        if len(document) > MAX_CHARS_PER_DOCUMENT:
            document = document[:MAX_CHARS_PER_DOCUMENT] + "..."
        context_parts.append(document.strip())
        context_parts.append("-" * 60)

    return "\n".join(context_parts)


def main():
    """Command-line search for quick retrieval checks."""
    parser = argparse.ArgumentParser(description="Search the NASA mission ChromaDB collection")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--mission", default="all", choices=["all", *MISSION_LABELS],
                        help="Only search one mission")
    args = parser.parse_args()

    collection, success, error = initialize_rag_system(args.chroma_dir, args.collection_name)
    if not success:
        raise SystemExit(f"Could not open collection: {error}")

    results = retrieve_documents(collection, args.question, args.top_k, args.mission)
    print(format_context(results["documents"][0], results["metadatas"][0], results["distances"][0]))


if __name__ == "__main__":
    main()
