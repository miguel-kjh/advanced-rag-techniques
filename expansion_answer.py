from pypdf import PdfReader
import os
import openai
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from helper_utils import word_wrap
from helper_utils import extract_text_from_pdf

FOLDER = "data"
PDF_FILE = "microsoft-annual-report.pdf"
PDF_PATH = os.path.join(FOLDER, PDF_FILE)

#main

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
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

    embedding_function = SentenceTransformerEmbeddingFunction(device="cuda")
    print(embedding_function([token_split_texts[0]]))

if __name__ == "__main__":
    main()