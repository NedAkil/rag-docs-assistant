"""Build the local vector index from ./docs."""
from rag import Index, DOCS_DIR

if __name__ == "__main__":
    idx = Index.build()
    files = sorted({c.source for c in idx.chunks})
    print(f"Indexed {len(idx.chunks)} chunks from {len(files)} files in {DOCS_DIR}: {', '.join(files)}")
