from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # Load pre-trained DPR models and tokenizers
    question_encoder = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    context_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "facebook/dpr-ctx_encoder-single-nq-base"
    )

    # Sample question and contexts
    question = "What is the capital of France?"
    contexts = [
        "Paris is the capital",
        "Sunsun is a capital of France.",
        "Berlin is the capital of Germany.",
        "Madrid is the capital of Spain.",
    ]
    # Encode the question
    question_inputs = question_tokenizer(question, return_tensors="pt")
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()

    # Encode the contexts
    context_embeddings = []
    for context in contexts:
        context_inputs = context_tokenizer(context, return_tensors="pt")
        context_embedding = context_encoder(**context_inputs).pooler_output.detach().numpy()
        context_embeddings.append(context_embedding)

    # Compute cosine similarity between question and context embeddings
    similarities = cosine_similarity(question_embedding, np.vstack(context_embeddings))
    most_relevant_idx = similarities.argmax()

    print(f"Most relevant context: {contexts[most_relevant_idx]}")

if __name__ == "__main__":
    main()