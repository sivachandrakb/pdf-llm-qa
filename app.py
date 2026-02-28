import streamlit as st
import PyPDF2
from rag_pipeline import RAGPipeline
from uncertainty import calculate_entropy, confidence_from_entropy

st.title("🧠 Uncertainty-Aware RAG System")

rag = RAGPipeline()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    chunks = text.split(". ")
    rag.build_index(chunks)

    st.success("PDF Indexed Successfully!")

    question = st.text_input("Ask a question:")

    if question:
        retrieved = rag.retrieve(question)
        context = " ".join(retrieved)

        answer, outputs = rag.generate_answer(context, question)

        entropy = calculate_entropy(outputs.scores)
        confidence = confidence_from_entropy(entropy)

        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📊 Uncertainty Metrics")
        st.write(f"Entropy: {entropy:.4f}")
        st.write(f"Confidence Score: {confidence:.4f}")

        st.subheader("📚 Retrieved Context")
        st.write(retrieved)
