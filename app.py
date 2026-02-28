import streamlit as st
from src.loader import load_and_split
from src.retriever import build_vectorstore
from src.llm import load_llm
from src.qa_engine import ask_question

st.set_page_config(page_title="PDF LLM QA")

st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded!")

    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF..."):
            chunks = load_and_split("temp.pdf")
            st.session_state.vectorstore = build_vectorstore(chunks)
        st.success("PDF Ready!")

    if "llm" not in st.session_state:
        with st.spinner("Loading LLM..."):
            st.session_state.llm = load_llm()

    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Generating answer..."):
            answer = ask_question(
                question,
                st.session_state.vectorstore,
                st.session_state.llm
            )

        st.markdown("### 🤖 Answer")
        st.write(answer)
