from transformers import pipeline

def load_llm():
    return pipeline(
        "text-generation",
        model="microsoft/phi-2",
        max_new_tokens=512
    )
