from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def build_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embedding_model)
