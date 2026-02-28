def ask_question(question, vectorstore, pipe):

    docs = vectorstore.similarity_search(question, k=3)

    context = ""
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        context += f"(Page {page})\n{doc.page_content}\n\n"

    prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If answer not found, say:
Answer not found in document.

Answer:
"""

    result = pipe(prompt)[0]["generated_text"]
    return result
