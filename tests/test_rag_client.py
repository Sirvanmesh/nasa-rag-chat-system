"""Smoke test for rag_client.py using a small temporary ChromaDB collection.

Run from the project folder:  python tests/test_rag_client.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag_client  # noqa: E402

SAMPLE_CHUNKS = [
    ("Houston, we've had a problem. Oxygen tank number 2 in the service module failed.",
     {"mission": "apollo_13", "source": "AS13_TEC", "document_category": "technical"}),
    ("The crew used the lunar module Aquarius as a lifeboat to return to Earth.",
     {"mission": "apollo_13", "source": "AS13_PAO", "document_category": "public_affairs_officer"}),
    ("The Eagle has landed. Tranquility Base here.",
     {"mission": "apollo_11", "source": "a11transcript_cm", "document_category": "command_module"}),
    ("Challenger, go at throttle up. Roger, go at throttle up.",
     {"mission": "challenger", "source": "107-AAG_STS-51L", "document_category": "mission_audio"}),
    # Exact duplicate of the first chunk, to check that duplicates are removed.
    ("Houston, we've had a problem. Oxygen tank number 2 in the service module failed.",
     {"mission": "apollo_13", "source": "AS13_CM", "document_category": "command_module"}),
]


def build_test_collection(chroma_dir: Path) -> None:
    """Create a small collection with real OpenAI embeddings."""
    client = rag_client.get_chroma_client(chroma_dir)
    collection = client.create_collection(
        name="test_missions",
        embedding_function=None,
        metadata={"embedding_model": rag_client.DEFAULT_EMBEDDING_MODEL},
        configuration={"hnsw": {"space": "cosine"}},
    )
    texts = [text for text, _ in SAMPLE_CHUNKS]
    embeddings = rag_client.get_openai_client().embeddings.create(
        model=rag_client.DEFAULT_EMBEDDING_MODEL, input=texts)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(texts))],
        documents=texts,
        metadatas=[metadata for _, metadata in SAMPLE_CHUNKS],
        embeddings=[item.embedding for item in embeddings.data],
    )


def main():
    temp_root = Path(tempfile.mkdtemp())
    chroma_dir = temp_root / "chroma_db_test"
    try:
        build_test_collection(chroma_dir)

        backends = rag_client.discover_chroma_backends(temp_root)
        print("1. Discovered backends:", [b["display_name"] for b in backends.values()])
        assert len(backends) == 1

        collection, success, error = rag_client.initialize_rag_system(str(chroma_dir), "test_missions")
        print("2. Initialize:", "OK" if success else error)
        assert success

        question = "What failed on Apollo 13?"
        results = rag_client.retrieve_documents(collection, question, n_results=3)
        documents = results["documents"][0]
        print(f"3. Top-3 for '{question}': {len(documents)} unique chunks (duplicate removed: "
              f"{len(documents) == len(set(documents))})")
        assert len(documents) == len(set(documents))
        assert results["distances"][0] == sorted(results["distances"][0])

        filtered = rag_client.retrieve_documents(collection, question, n_results=3,
                                                 mission_filter="apollo_11")
        missions = {m["mission"] for m in filtered["metadatas"][0]}
        print("4. Mission filter apollo_11 returns missions:", missions)
        assert missions == {"apollo_11"}

        print("5. Formatted context:\n")
        print(rag_client.format_context(documents, results["metadatas"][0], results["distances"][0]))
        print("\nAll rag_client checks passed.")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
