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

def generate_answer(self, context, question):
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = self.tokenizer(prompt, return_tensors="pt")

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=self.tokenizer.eos_token_id,
        pad_token_id=self.tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True
    )

    generated_text = self.tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True
    )

    # Extract only answer part
    answer = generated_text.split("Answer:")[-1].strip()

    return answer, outputs
