"""Streamlit chat UI: upload docs, ask questions, see sources."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import rag  # noqa: E402

st.set_page_config(page_title="Docs Assistant", page_icon="📚")
st.title("📚 Ask your documents")
st.caption("RAG assistant: hybrid retrieval (BM25 + embeddings), cited sources, refuses when the docs don't say.")


@st.cache_resource
def get_index(version: int):
    return rag.Index.build()


if "version" not in st.session_state:
    st.session_state.version, st.session_state.chat = 0, []

with st.sidebar:
    st.header("Documents")
    ups = st.file_uploader("Upload PDF / Markdown / TXT", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if st.button("Add to index") and ups:
        for u in ups:
            (rag.DOCS_DIR / Path(u.name).name).write_bytes(u.getvalue())
        st.session_state.version += 1
        st.success(f"Added {len(ups)} file(s).")
    index = get_index(st.session_state.version)
    st.write(f"{len(index.chunks)} chunks from " + ", ".join(sorted({c.source for c in index.chunks})))
    try:
        st.caption(f"LLM provider: {rag.llm_provider()}")
    except RuntimeError as e:
        st.error(str(e))

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for s in m.get("sources", []):
            with st.expander(f"[{s['n']}] {s['source']} - page {s['page']}"):
                st.write(s["text"])

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                r = rag.answer(index, q)
            except Exception as e:  # noqa: BLE001
                r = {"answer": f"Error: {e}", "sources": []}
        st.markdown(r["answer"])
        for s in r["sources"]:
            with st.expander(f"[{s['n']}] {s['source']} - page {s['page']}"):
                st.write(s["text"])
    st.session_state.chat.append({"role": "assistant", "content": r["answer"], "sources": r["sources"]})
