"""Core RAG pipeline: load -> chunk -> embed -> hybrid retrieve -> answer with citations."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
INDEX_PATH = ROOT / "index" / "index.npz"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NOT_FOUND = "I could not find this in the provided documents."


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    heading: str = ""


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def load_pages(path: Path) -> list[tuple[int, str]]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
    if ext in (".md", ".txt"):
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    return []


def chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[tuple[str, str]]:
    """Split by markdown headings first, then by sliding window. Returns (heading, chunk)."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))
    out = []
    for h, body in sections:
        body = body.strip()
        if not body:
            continue
        start = 0
        while start < len(body):
            out.append((h, body[start:start + size]))
            if start + size >= len(body):
                break
            start += size - overlap
    return out


def build_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for f in sorted(docs_dir.glob("*")):
        for page, text in load_pages(f):
            for h, c in chunk_text(text):
                prefix = f"{h}: " if h else ""
                chunks.append(Chunk(prefix + c, f.name, page, h))
    return chunks


_model = None


def embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


class Index:
    def __init__(self, chunks: list[Chunk], vecs: np.ndarray):
        self.chunks, self.vecs = chunks, vecs
        self.bm25 = BM25Okapi([_tok(c.text) for c in chunks])

    @classmethod
    def build(cls, docs_dir: Path = DOCS_DIR, save: bool = True) -> "Index":
        chunks = build_chunks(docs_dir)
        if not chunks:
            raise ValueError(f"No documents found in {docs_dir}")
        vecs = embedder().encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False)
        if save:
            INDEX_PATH.parent.mkdir(exist_ok=True)
            np.savez(INDEX_PATH, vecs=vecs, text=[c.text for c in chunks],
                     source=[c.source for c in chunks], page=[c.page for c in chunks],
                     heading=[c.heading for c in chunks])
        return cls(chunks, vecs)

    @classmethod
    def load(cls) -> "Index":
        d = np.load(INDEX_PATH, allow_pickle=False)
        chunks = [Chunk(str(t), str(s), int(p), str(h))
                  for t, s, p, h in zip(d["text"], d["source"], d["page"], d["heading"])]
        return cls(chunks, d["vecs"])

    def search(self, query: str, k: int = 4, rrf_k: int = 60) -> list[tuple[Chunk, float]]:
        """Hybrid retrieval: reciprocal-rank fusion of BM25 and cosine similarity.
        Returned score is the cosine similarity (used for the refusal threshold)."""
        qv = embedder().encode([query], normalize_embeddings=True)[0]
        cos = self.vecs @ qv
        bm = np.array(self.bm25.get_scores(_tok(query)))
        fused = np.zeros(len(self.chunks))
        for scores in (cos, bm):
            for rank, i in enumerate(np.argsort(-scores)):
                fused[i] += 1.0 / (rrf_k + rank + 1)
        top = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(cos[i])) for i in top]


# ---------------- LLM (pluggable) ----------------

SYSTEM = (
    "You answer questions strictly from the provided context excerpts. "
    "Cite sources inline like [1], [2] using the excerpt numbers. "
    f"If the context does not contain the answer, reply exactly: {NOT_FOUND} "
    "Never use outside knowledge. Be concise."
)


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        load_dotenv(Path.home() / ".gemini" / ".env")  # optional local fallback
    except ImportError:
        pass


def llm_provider() -> str:
    _load_env()
    for name, var in (("gemini", "GEMINI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")):
        if os.getenv(var):
            return name
    raise RuntimeError("Set GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY (see .env.example).")


def complete(prompt: str, system: str = SYSTEM, retries: int = 6) -> str:
    """Call the LLM, backing off on rate limits (e.g. Gemini free tier = 5 req/min)."""
    import time
    for attempt in range(retries):
        try:
            return _complete(prompt, system)
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1 or not any(t in str(e) for t in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "overloaded")):
                raise
            time.sleep(15 + 10 * attempt)


def _complete(prompt: str, system: str) -> str:
    p = llm_provider()
    if p == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        r = client.models.generate_content(
            model=os.getenv("LLM_MODEL", "gemini-3-flash-preview"), contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0))
        return (r.text or "").strip()
    if p == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5"), max_tokens=800, system=system,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text.strip()
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()


MIN_SIM = float(os.getenv("MIN_SIMILARITY", "0.25"))


def answer(index: Index, question: str, k: int = 4) -> dict:
    hits = index.search(question, k=k)
    if not hits or max(s for _, s in hits) < MIN_SIM:  # cheap guard before calling the LLM
        return {"answer": NOT_FOUND, "sources": [], "hits": hits}
    ctx = "\n\n".join(f"[{i+1}] ({c.source}, p.{c.page})\n{c.text}" for i, (c, _) in enumerate(hits))
    text = complete(f"Context:\n{ctx}\n\nQuestion: {question}")
    refused = NOT_FOUND.lower().rstrip(".") in text.lower()
    used = [] if refused else [
        {"n": i + 1, "source": c.source, "page": c.page, "text": c.text}
        for i, (c, _) in enumerate(hits) if f"[{i+1}]" in text] or \
        [{"n": i + 1, "source": c.source, "page": c.page, "text": c.text} for i, (c, _) in enumerate(hits)]
    return {"answer": text, "sources": used, "hits": hits}
