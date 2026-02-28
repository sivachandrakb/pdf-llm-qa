import streamlit as st
import PyPDF2
from rag_pipeline import RAGPipeline
from uncertainty import calculate_entropy, confidence_from_entropy

st.set_page_config(page_title="U-RAG", layout="wide")
st.title("🧠 Uncertainty-Aware RAG System")

# ------------------------
# SESSION STATE INITIALIZE
# ------------------------

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# FILE UPLOAD
# ------------------------

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.indexed:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    chunks = text.split(". ")
    st.session_state.rag.build_index(chunks)
    st.session_state.indexed = True
    st.success("PDF Indexed Successfully!")

# ------------------------
# CHAT INTERFACE
# ------------------------

if st.session_state.indexed:

    question = st.chat_input("Ask a question about the document...")

    if question:
        retrieved = st.session_state.rag.retrieve(question)
        context = " ".join(retrieved)

        answer, outputs = st.session_state.rag.generate_answer(context, question)

        entropy = calculate_entropy(outputs.scores)
        confidence = confidence_from_entropy(entropy)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "entropy": entropy,
            "confidence": confidence
        })

# ------------------------
# DISPLAY CHAT HISTORY
# ------------------------

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])
        st.caption(f"Entropy: {chat['entropy']:.4f} | Confidence: {chat['confidence']:.4f}")
