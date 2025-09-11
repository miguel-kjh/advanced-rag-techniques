from typing import List
from pypdf import PdfReader
import os
from openai import OpenAI
import chromadb
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import umap
import matplotlib.pyplot as plt

from helper_utils import project_embeddings, word_wrap

FOLDER   = "data"
PDF_FILE = "microsoft-annual-report.pdf"
PDF_PATH = os.path.join(FOLDER, PDF_FILE)
BD_PATH  = os.path.join(FOLDER, "chroma_db")
COLLECTION_NAME = "annual_report"

TOP_K = 20
TOP_MOST_RELEVANT = 10
SEED  = 123

def generate_answer(query: str, context: str, client: OpenAI, model: str = "gpt-4o") -> str:
    system_prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    Provide a concise and accurate answer based on the provided context from the document. 
    If the context does not contain relevant information, respond with "The provided context does not contain information related to the question."
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
        ],
        temperature=0,
        seed=SEED,
    )
    answer = response.choices[0].message.content
    return answer

def create_db(texts: List[str], client: chromadb.PersistentClient, embedding_function: SentenceTransformerEmbeddingFunction) -> chromadb.api.models.Collection:
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    collection.add(documents=texts, ids=[str(i) for i in range(len(texts))])
    return collection

def chunking_data() -> List[str]:
    reader = PdfReader(PDF_PATH)
    pdf_texts = [
        p.extract_text().strip() for p in reader.pages
    ]
    # filter out empty pages
    pdf_texts = [p for p in pdf_texts if p]

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )

    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, 
    )
    
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    print(f"Number of character-based chunks: {len(character_split_texts)}")
    print(f"Number of token-based chunks: {len(token_split_texts)}")
    return token_split_texts

#main
def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_key)
    client = chromadb.PersistentClient(path=BD_PATH)
    existing = [c.name for c in client.list_collections()]
    print(f"Existing collections: {existing}")
    embedding_function = SentenceTransformerEmbeddingFunction(device="cuda")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

    if COLLECTION_NAME in existing:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        print(f"Existing collection named {COLLECTION_NAME} found.")
    else:
        token_split_texts = chunking_data()
        collection = create_db(token_split_texts, client, embedding_function)
        print("Collection created and data added.")
    

    # count
    print(f"Number of documents in the collection: {collection.count()}")

    query = "What was the total profit for the year, and how does it compare to the previous year?"
    results = collection.query(
        query_texts=query,
        n_results=TOP_K,
        include=["documents", "embeddings", "distances"]
    )
    retrieved_documents = results["documents"][0]
    for i, doc in enumerate(retrieved_documents):
        print(f"\nDocument {i+1}:\n{word_wrap(doc, width=100)}")

    # Reranking with Cross-Encoder
    pairs = [[query, doc] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)
    # sort documents by scores
    doc_score = list(zip(retrieved_documents, scores))
    doc_score = sorted(doc_score, key=lambda x: x[1], reverse=True)
    # Get the top most relevant documents
    top_docs = [doc for doc, score in doc_score[:TOP_MOST_RELEVANT]]
    print(f"\nTop {TOP_MOST_RELEVANT} most relevant documents after reranking:")
    for i, doc in enumerate(top_docs):
        print(f"\nDocument {i+1}:\n{word_wrap(doc, width=100)}")

    # Generate answer based on the top most relevant documents
    context = "\n\n".join(top_docs)
    answer = generate_answer(query, context, openai_client)
    print(f"\nAnswer:\n{word_wrap(answer, width=100)}")

if __name__ == "__main__":
    main()