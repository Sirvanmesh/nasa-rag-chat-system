"""Check the chunking rules on every mission file. No API calls.

Run from the project folder:  python tests/test_chunking.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from embedding_pipeline import clean_text, split_text  # noqa: E402

SETTINGS = [(500, 100), (1000, 200), (300, 0), (200, 150)]


def check_file(text: str, chunk_size: int, chunk_overlap: int) -> int:
    """Assert the chunking rules for one text and return the number of chunks."""
    spans = split_text(text, chunk_size, chunk_overlap)

    assert spans[0][0] == 0 and spans[-1][1] == len(text), "chunks must cover the whole text"
    for start, end in spans:
        assert 0 < end - start <= chunk_size, f"chunk of {end - start} chars > {chunk_size}"
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
        assert previous_end - next_start == chunk_overlap, "overlap must be exactly chunk_overlap"
    return len(spans)


def main():
    files = sorted((PROJECT_DIR / "data_text").glob("*/*.txt"))
    texts = {path.name: clean_text(path.read_text(encoding="utf-8")) for path in files}

    for chunk_size, chunk_overlap in SETTINGS:
        total = sum(check_file(text, chunk_size, chunk_overlap) for text in texts.values())
        print(f"chunk_size={chunk_size:<5} overlap={chunk_overlap:<4} -> {total:>6} chunks "
              f"across {len(texts)} files: all checks passed")

    assert split_text("short text", 500, 100) == [(0, 10)], "short text should be one chunk"
    print("Short text stays one chunk: passed")


if __name__ == "__main__":
    main()
