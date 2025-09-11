from typing import List
from pypdf import PdfReader
import os
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import umap
import matplotlib.pyplot as plt

from helper_utils import project_embeddings, word_wrap
from helper_utils import extract_text_from_pdf

FOLDER   = "data"
PDF_FILE = "microsoft-annual-report.pdf"
PDF_PATH = os.path.join(FOLDER, PDF_FILE)
BD_PATH  = os.path.join(FOLDER, "chroma_db")
COLLECTION_NAME = "annual_report"
NUMBER_QUERIES_EXPANSION = 5

TOP_K = 5
SEED  = 123

def multi_query_generation(query: str, client: OpenAI, number_queries_expansion: int = NUMBER_QUERIES_EXPANSION, model: str = "gpt-4o") -> str:
    system_prompt = f"""
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to {number_queries_expansion} related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        seed=SEED,
    )
    content = response.choices[0].message.content
    queries = content.split("\n")
    queries.append(query)  # include original query
    return queries

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
    queries = multi_query_generation(query, openai_client)

    results = collection.query(
        query_texts=queries,
        n_results=TOP_K,
        include=["documents", "embeddings", "distances"]
    )
    retrieved_documents = results["documents"]
    
    #quitamos duplicados
    unique_docs = set()
    for doc in retrieved_documents:
        for d in doc:
            unique_docs.add(d)
    print(f"\nRetrieved unique documents {len(unique_docs)}:")
    
    
    embeddings = collection.get(include=["embeddings"])["embeddings"]
    umap_transform = umap.UMAP(random_state=SEED, transform_seed=SEED).fit(embeddings)
    projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

    retrieved_embeddings = results["embeddings"]
    result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]
    original_query_embedding = embedding_function([query])
    augmented_query_embedding = embedding_function(queries)

    projected_original_query_embedding = project_embeddings(
        original_query_embedding, umap_transform
    )
    projected_augmented_query_embedding = project_embeddings(
        augmented_query_embedding, umap_transform
    )
    projected_retrieved_embeddings = project_embeddings(
        result_embeddings, umap_transform
    )

    # Plot the projected query and retrieved documents in the embedding space
    plt.figure()

    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
    )
    plt.scatter(
        projected_retrieved_embeddings[:, 0],
        projected_retrieved_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="g",
    )
    plt.scatter(
        projected_original_query_embedding[:, 0],
        projected_original_query_embedding[:, 1],
        s=150,
        marker="X",
        color="r",
    )
    plt.scatter(
        projected_augmented_query_embedding[:, 0],
        projected_augmented_query_embedding[:, 1],
        s=150,
        marker="X",
        color="orange",
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"{query}")
    plt.axis("off")
    plt.show()  # display the plot

if __name__ == "__main__":
    main()