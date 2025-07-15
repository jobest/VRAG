import streamlit as st
import asyncio
from dotenv import load_dotenv
from config import load_config
from pdf_utils import process_pdf
from qdrant_utils import setup_qdrant, store_embeddings
from voice_agent import process_query

def init_session_state():
    defaults = {
        "initialized": False,
        "setup_complete": False,
        "pdf_uploaded_this_session": False,
        "processed_documents": [],
        "client": None,
        "embedding_model": None,
        "selected_voice": "coral"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

async def main():
    st.set_page_config(page_title="Voice RAG Agent", page_icon="🎙️", layout="wide")
    st.title("🎙️ Voice RAG Agent")
    st.markdown("Upload a PDF, ask a question, and hear the answer!")

    init_session_state()
    config = load_config()

    voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
    st.session_state.selected_voice = st.selectbox("Choose Voice", voices, index=voices.index("coral"))

    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])

    if uploaded_file and not st.session_state.pdf_uploaded_this_session:
        st.session_state.pdf_uploaded_this_session = True
        st.session_state.setup_complete = False

    if st.session_state.pdf_uploaded_this_session and not st.session_state.setup_complete:
        with st.spinner("Processing PDF and initializing Qdrant..."):
            try:
                client, embedding_model = setup_qdrant(config["QDRANT_URL"], config["QDRANT_API_KEY"])
                docs = process_pdf(uploaded_file)
                store_embeddings(client, embedding_model, docs)
                st.session_state.client = client
                st.session_state.embedding_model = embedding_model
                st.session_state.processed_documents = docs
                st.session_state.setup_complete = True
                st.success("PDF processed and stored in vector DB!")
            except Exception as e:
                st.error(f"Failed to process file: {e}")
                st.session_state.pdf_uploaded_this_session = False
                st.session_state.setup_complete = False

    if st.session_state.setup_complete:
        query = st.text_input("Ask a question about the document:")
        if st.button("Ask Agent") and query:
            await process_query(
                query,
                st.session_state.client,
                st.session_state.embedding_model,
                config["OPENAI_API_KEY"],
                st.session_state.selected_voice
            )

if __name__ == "__main__":
    asyncio.run(main())