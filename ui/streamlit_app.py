import os
import time
from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
REQUEST_TIMEOUT = int(os.getenv("UI_REQUEST_TIMEOUT", "240"))
SUPPORTED_UPLOADS = ["txt", "md", "pdf"]


st.set_page_config(page_title="On-Prem RAG Agent", page_icon="RAG", layout="wide")


def main() -> None:
    _init_state()
    _render_header()
    stats = _safe_get("/stats")
    health = _safe_get("/health")
    _render_sidebar(health, stats)
    _render_chat()


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("debug_enabled", False)
    st.session_state.setdefault("top_k", 5)


def _render_header() -> None:
    left, right = st.columns([0.7, 0.3], vertical_alignment="center")
    with left:
        st.title("On-Prem RAG Agent")
    with right:
        st.caption(f"API: `{API_URL}`")


def _render_sidebar(health: dict[str, Any] | None, stats: dict[str, Any] | None) -> None:
    with st.sidebar:
        st.header("Controls")
        if health:
            status = health.get("status", "unknown")
            backend = health.get("backend", "unknown")
            st.success(f"{status.upper()} · {backend}")
        else:
            st.error("API unavailable")

        st.session_state.top_k = st.slider("Sources", min_value=1, max_value=20, value=st.session_state.top_k)
        st.session_state.debug_enabled = st.toggle("Debug trace", value=st.session_state.debug_enabled)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Warm Up", use_container_width=True):
                _run_action("POST", "/warmup", "Warm-up")
        with col_b:
            if st.button("Ingest", use_container_width=True):
                _run_action("POST", "/ingest", "Ingestion")

        uploaded_files = st.file_uploader(
            "Add documents",
            type=SUPPORTED_UPLOADS,
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Save Uploaded Files", use_container_width=True):
            saved = _save_uploads(uploaded_files)
            st.success(f"Saved {saved} file(s) to `{RAW_DATA_DIR}`")

        st.divider()
        _render_stats(stats)


def _render_stats(stats: dict[str, Any] | None) -> None:
    st.subheader("Stats")
    if not stats:
        st.caption("Stats unavailable.")
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("Documents", stats.get("total_documents", 0))
    metric_cols[1].metric("Chunks", stats.get("total_chunks", 0))
    metric_cols[2].metric("Backend", stats.get("backend", "unknown"))

    vector_store = stats.get("vector_store", {})
    if vector_store.get("backend") == "qdrant":
        st.caption(f"Qdrant points: `{vector_store.get('points', 0)}`")
    elif vector_store.get("backend") == "faiss":
        st.caption(f"FAISS metadata: `{vector_store.get('loaded_metadata', 0)}`")

    with st.expander("Model Settings"):
        _render_key_values(
            {
                "Embedding": stats.get("embedding_model"),
                "Reranker": stats.get("reranker_model"),
                "LLM": stats.get("ollama_model"),
                "Provider": stats.get("llm_provider"),
            }
        )

    with st.expander("Vector Store"):
        _render_key_values(_friendly_vector_store(vector_store))

    with st.expander("Ingestion"):
        _render_key_values(_friendly_ingestion(stats.get("ingestion", {})))

    with st.expander("Retrieval"):
        _render_key_values(_friendly_retrieval(stats.get("retrieval", {})))

    with st.expander("Raw stats"):
        st.json(stats)


def _render_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])
            if message.get("debug"):
                _render_debug(message["debug"])

    prompt = st.chat_input("Ask a question about the ingested documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            started = time.perf_counter()
            response = _post_json(
                "/query",
                {
                    "question": prompt,
                    "top_k": st.session_state.top_k,
                    "debug": st.session_state.debug_enabled,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

        if response is None:
            answer = "The query failed. Check the API logs for details."
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        answer = response.get("answer", "")
        sources = response.get("sources", [])
        debug = response.get("debug")
        st.markdown(answer)
        st.caption(f"`{response.get('backend', 'unknown')}` · {elapsed_ms:.0f} ms")
        _render_sources(sources)
        if debug:
            _render_debug(debug)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "debug": debug,
        }
    )


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=True):
        st.dataframe(
            [
                {
                    "source": source.get("source"),
                    "chunk_id": source.get("chunk_id"),
                    "score": round(float(source.get("score", 0.0)), 4),
                }
                for source in sources
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_debug(debug: dict[str, Any]) -> None:
    with st.expander("Debug trace"):
        retrieval = debug.get("retrieval", {})
        timings = retrieval.get("timings_ms", {})
        if timings:
            cols = st.columns(5)
            cols[0].metric("Embed", f"{timings.get('embedding', 0)} ms")
            cols[1].metric("Dense", f"{timings.get('dense', 0)} ms")
            cols[2].metric("Sparse", f"{timings.get('sparse', 0)} ms")
            cols[3].metric("Fusion", f"{timings.get('fusion', 0)} ms")
            cols[4].metric("Total", f"{timings.get('total', 0)} ms")

        tabs = st.tabs(["Fused", "Dense", "Sparse", "Reranked", "Raw"])
        with tabs[0]:
            _render_debug_table(retrieval.get("fused_results", []))
        with tabs[1]:
            _render_debug_table(retrieval.get("dense_results", []))
        with tabs[2]:
            _render_debug_table(retrieval.get("sparse_results", []))
        with tabs[3]:
            _render_debug_table(debug.get("reranked_results", []))
        with tabs[4]:
            st.json(debug)


def _render_debug_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.caption("No rows.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _safe_get(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(str(exc))
        return None


def _run_action(method: str, path: str, label: str) -> None:
    try:
        with st.spinner(f"{label} running..."):
            if method == "POST":
                response = requests.post(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
            else:
                response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"{label} failed: {exc}")
        return
    payload = response.json()
    st.success(f"{label} completed")
    _render_action_result(payload)


def _render_action_result(payload: dict[str, Any]) -> None:
    if {"changed_files", "skipped_files", "deleted_files"}.issubset(payload):
        st.caption(
            " · ".join(
                [
                    f"documents: `{payload.get('total_documents', payload.get('documents_ingested', 0))}`",
                    f"chunks: `{payload.get('total_chunks', payload.get('chunks_created', 0))}`",
                    f"changed: `{payload.get('changed_files', 0)}`",
                    f"skipped: `{payload.get('skipped_files', 0)}`",
                    f"deleted: `{payload.get('deleted_files', 0)}`",
                ]
            )
        )
        return
    if "warmed" in payload:
        warmed = payload.get("warmed", {})
        st.caption(
            " · ".join(f"{name}: `{'yes' if value else 'no'}`" for name, value in warmed.items())
        )
        return
    st.caption(payload.get("status", "completed"))


def _render_key_values(items: dict[str, Any]) -> None:
    rows = [
        {"setting": _label(key), "value": _display_value(value)}
        for key, value in items.items()
        if value is not None
    ]
    if not rows:
        st.caption("No data.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _friendly_vector_store(vector_store: dict[str, Any]) -> dict[str, Any]:
    backend = vector_store.get("backend")
    if backend == "qdrant":
        return {
            "backend": "Qdrant",
            "collection": vector_store.get("collection"),
            "points": vector_store.get("points"),
            "url": vector_store.get("url"),
            "collection_exists": vector_store.get("collection_exists"),
        }
    if backend == "faiss":
        return {
            "backend": "FAISS",
            "loaded_metadata": vector_store.get("loaded_metadata"),
            "index_exists": vector_store.get("index_exists"),
            "index_path": vector_store.get("index_path"),
            "metadata_path": vector_store.get("metadata_path"),
        }
    return vector_store


def _friendly_ingestion(ingestion: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_exists": ingestion.get("manifest_exists"),
        "files": ingestion.get("files"),
        "total_manifest_chunks": ingestion.get("total_manifest_chunks"),
        "chunk_size": ingestion.get("chunk_size"),
        "chunk_overlap": ingestion.get("chunk_overlap"),
        "embedding_model": ingestion.get("embedding_model"),
    }


def _friendly_retrieval(retrieval: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_k_retrieval": retrieval.get("top_k_retrieval"),
        "top_k_reranked": retrieval.get("top_k_reranked"),
        "hybrid_alpha": retrieval.get("hybrid_alpha"),
        "rerank_threshold": retrieval.get("rerank_relative_threshold"),
        "bm25_documents": retrieval.get("bm25_documents"),
    }


def _label(value: str) -> str:
    return value.replace("_", " ").title()


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _save_uploads(uploaded_files: list[Any]) -> int:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for uploaded_file in uploaded_files:
        destination = RAW_DATA_DIR / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getbuffer())
        saved += 1
    return saved


if __name__ == "__main__":
    main()
