#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

import os
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

import ragas_evaluator
import rag_client
import llm_client

load_dotenv()

RAGAS_AVAILABLE = ragas_evaluator.RAGAS_AVAILABLE

MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4"]
MISSION_OPTIONS = {"all": "All missions", **rag_client.MISSION_LABELS}

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

if not RAGAS_AVAILABLE:
    st.warning("RAGAS not available. Install with: pip install ragas")


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""

    return rag_client.discover_chroma_backends()


@st.cache_resource
def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None


def format_context(documents: List[str], metadatas: List[Dict],
                   distances: Optional[List[float]] = None) -> str:
    """Format retrieved documents into context"""

    return rag_client.format_context(documents, metadatas, distances)


def generate_response(openai_key, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(openai_key, user_message, context, conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def score_color(score: float) -> str:
    """Traffic-light label for a 0-1 score."""
    if score >= 0.8:
        return "🟢"
    if score >= 0.6:
        return "🟠"
    return "🔴"


def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")

    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(
                label=f"{score_color(score)} {metric_name.replace('_', ' ').title()}",
                value=f"{score:.3f}",
                delta=None
            )
            st.sidebar.progress(min(max(float(score), 0.0), 1.0))

    if scores.get("failed_metrics"):
        st.sidebar.caption(f"Not scored: {scores['failed_metrics']}")


def display_inline_scores(scores: Dict[str, float]):
    """Show the RAGAS scores of one answer directly below it."""
    if "error" in scores:
        st.caption(f"⚠️ Evaluation: {scores['error']}")
        return

    numeric_scores = {name: value for name, value in scores.items() if isinstance(value, (int, float))}
    columns = st.columns(max(len(numeric_scores), 1))
    for column, (name, value) in zip(columns, numeric_scores.items()):
        column.metric(f"{score_color(value)} {name.replace('_', ' ').title()}", f"{value:.2f}")
    if scores.get("failed_metrics"):
        st.caption(f"Not scored: {scores['failed_metrics']}")


def display_sources(sources: List[Dict]):
    """List the retrieved chunks behind one answer in a collapsed section."""
    with st.expander(f"📚 Sources ({len(sources)})"):
        for number, source in enumerate(sources, start=1):
            st.markdown(f"**[Source {number}]** {source['mission']} · `{source['source']}` · "
                        f"relevance {source['relevance']:.2f}")
            st.caption(source["preview"])


def display_message(message: Dict):
    """Render one chat message, plus its sources and scores for assistant answers."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            display_sources(message["sources"])
        if message.get("evaluation"):
            display_inline_scores(message["evaluation"])


def summarize_sources(docs_result: Dict) -> List[Dict]:
    """Keep short source details for display in the chat history."""
    return [
        {
            "mission": rag_client.format_mission_name(metadata.get("mission", "unknown")),
            "source": metadata.get("source", "unknown"),
            "relevance": 1 - distance,
            "preview": document[:300].strip() + ("..." if len(document) > 300 else ""),
        }
        for document, metadata, distance in zip(docs_result["documents"][0],
                                                docs_result["metadatas"][0],
                                                docs_result["distances"][0])
    ]


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # Discover available backends
        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Please run the embedding pipeline first:\n`python embedding_pipeline.py`")
            st.stop()

        # Backend selection
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}

        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )

        selected_backend = available_backends[selected_backend_key]

        # API Key input
        st.subheader("🔑 OpenAI Settings")
        # The .env key is never put into the text box, so it is never sent to the browser.
        env_key = os.getenv("OPENAI_API_KEY", "")
        key_override = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            value="",
            help="Leave empty to use OPENAI_API_KEY from your .env file."
        )
        openai_key = key_override or env_key

        if not openai_key:
            st.warning("Please enter your OpenAI API key or add it to .env")
            st.stop()
        elif key_override:
            # rag_client and ragas_evaluator read the key from the environment.
            os.environ["OPENAI_API_KEY"] = key_override
        else:
            st.caption("✅ Using the API key from .env")

        # Model selection
        model_choice = st.selectbox(
            "OpenAI Model",
            options=MODEL_OPTIONS,
            help="Choose the OpenAI model for responses"
        )

        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        mission_choice = st.selectbox(
            "Mission",
            options=list(MISSION_OPTIONS.keys()),
            format_func=lambda x: MISSION_OPTIONS[x],
            help="Only search documents from this mission"
        )
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)

        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE,
                                        disabled=not RAGAS_AVAILABLE)

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.session_state.last_contexts = []
            st.rerun()

        # Initialize RAG system when backend changes
        if (st.session_state.current_backend != selected_backend_key):
            st.session_state.current_backend = selected_backend_key
            # Clear cache to force reinitialization
            st.cache_resource.clear()

    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):

        collection, success, error = initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )

    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    # Display evaluation metrics if available
    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

    # Display chat messages
    for message in st.session_state.messages:
        display_message(message)

    # Chat input
    if prompt := st.chat_input("Ask about NASA space missions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message({"role": "user", "content": prompt})

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Retrieve relevant documents
                docs_result = retrieve_documents(
                    collection,
                    prompt,
                    n_docs,
                    mission_choice
                )

                # Format context
                context = ""
                contexts_list = []
                sources = []
                if docs_result and docs_result.get("documents") and docs_result["documents"][0]:
                    context = format_context(docs_result["documents"][0], docs_result["metadatas"][0],
                                             docs_result["distances"][0])
                    contexts_list = docs_result["documents"][0]
                    sources = summarize_sources(docs_result)
                st.session_state.last_contexts = contexts_list

                # Generate response
                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model_choice
                )
                st.markdown(response)

            # Evaluate response quality if enabled
            evaluation_scores = None
            if enable_evaluation and RAGAS_AVAILABLE:
                with st.spinner("Evaluating response quality..."):
                    evaluation_scores = evaluate_response_quality(
                        prompt,
                        response,
                        contexts_list
                    )
                    st.session_state.last_evaluation = evaluation_scores

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "evaluation": evaluation_scores,
        })
        st.rerun()


if __name__ == "__main__":
    main()
