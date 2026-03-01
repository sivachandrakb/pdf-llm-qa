# import faiss
# import numpy as np
# from embeddings import get_embeddings
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch 

# class RAGPipeline:
#     def __init__(self):
#         self.index = None
#         self.chunks = None

#         self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#         self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

#     def build_index(self, texts):
#         self.chunks = texts
#         embeddings = get_embeddings(texts)
#         dimension = embeddings.shape[1]

#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(np.array(embeddings))

#     def retrieve(self, query, k=3):
#         query_embedding = get_embeddings([query])
#         distances, indices = self.index.search(np.array(query_embedding), k)
#         return [self.chunks[i] for i in indices[0]]

#     def generate_answer(self, context, question):
#         prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#         inputs = self.tokenizer(prompt, return_tensors="pt")

#         outputs = self.model.generate(
#             **inputs,
#             max_length=200,
#             output_scores=True,
#             return_dict_in_generate=True
#         )

#         answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
#         return answer, outputs

import numpy as np
from embeddings import get_embeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity

class RAGPipeline:
    def __init__(self):
        self.chunks = None
        self.embeddings = None

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def build_index(self, texts):
        self.chunks = texts
        self.embeddings = get_embeddings(texts)

    def retrieve(self, query, k=3):
        query_embedding = get_embeddings([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_k = similarities.argsort()[0][-k:][::-1]
        return [self.chunks[i] for i in top_k]

    def generate_answer(self, context, question):
        prompt = f"""
Answer the question based only on the context.

Context:
{context}

Question:
{question}
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            output_scores=True,
            return_dict_in_generate=True
        )

        answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return answer, outputs
