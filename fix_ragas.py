"""Patch the VertexAI imports in RAGAS 0.4.3 so `import ragas` works with current LangChain.

RAGAS 0.4.3 imports ChatVertexAI and VertexAI from old langchain-community paths.
These classes now live in langchain-google-vertexai. Safe to run more than once.

Usage:  python fix_ragas.py
"""

import importlib.util
from pathlib import Path

REPLACEMENTS = {
    "from langchain_community.chat_models.vertexai import ChatVertexAI":
        "from langchain_google_vertexai import ChatVertexAI",
    "from langchain_community.llms import VertexAI":
        "from langchain_google_vertexai import VertexAI",
}


def main():
    # find_spec locates the package without importing it (the import is what fails).
    spec = importlib.util.find_spec("ragas")
    if spec is None or spec.origin is None:
        raise SystemExit("RAGAS is not installed. Run: pip install -r requirements.txt")

    base_file = Path(spec.origin).parent / "llms" / "base.py"
    source = base_file.read_text(encoding="utf-8")
    patched = source
    for old, new in REPLACEMENTS.items():
        patched = patched.replace(old, new)

    if patched == source:
        print(f"Already patched (nothing to change): {base_file}")
    else:
        base_file.write_text(patched, encoding="utf-8")
        print(f"Patched VertexAI imports in: {base_file}")


if __name__ == "__main__":
    main()
