# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# File/folder operations
import os

# Regex for cleaning noisy text
import re

# Load .txt files from a directory
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Text splitter (better for RAG pipelines)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HuggingFace embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# Vector database
from langchain_chroma import Chroma


# =========================================================
# CLEAN DOCUMENT TEXT
# =========================================================
#
# Why clean?
# Raw text often contains:
# - URLs
# - citations [123]
# - extra spaces/newlines
# - archived links
#
# These reduce retrieval quality, so we strip them out.

def clean_text(text):
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"

\[\d+\]

", "", text)       # remove citations like [123]
    text = re.sub(r"\s+", " ", text)          # normalize whitespace
    return text.strip()


# =========================================================
# STEP 1 -> LOAD DOCUMENTS
# =========================================================

def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} folder not found.")

    # Load all .txt files
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError("No txt files found.")

    # Clean content
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Print preview of first 2 docs
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)}")
        print(f"Preview:\n{doc.page_content[:300]}")
        print("-" * 50)

    return documents


# =========================================================
# STEP 2 -> SPLIT DOCUMENTS INTO CHUNKS
# =========================================================

def split_documents(documents, chunk_size=1500, chunk_overlap=100):
    print("\nSplitting documents into chunks...")

    # Why chunking?
    # LLMs struggle with very large inputs.
    # Breaking into smaller overlapping chunks
    # improves retrieval and preserves context.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    print(f"\nTotal chunks created: {len(chunks)}")

    # Print preview of first 10 chunks
    for i, chunk in enumerate(chunks[:10]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Chunk Length: {len(chunk.page_content)}")
        print(chunk.page_content[:500])
        print("-" * 50)

    return chunks


# =========================================================
# STEP 3 -> CREATE VECTOR DATABASE
# =========================================================

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("\nCreating embeddings and storing in ChromaDB...")

    # Embedding model: converts text → vectors
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB stores chunks + vectors
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}  # cosine similarity
    )

    print("\nVector store created successfully.")
    return vectorstore


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():
    print("=== RAG INGESTION PIPELINE ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # If DB already exists, load it
    if os.path.exists(persistent_directory):
        print("Vector store already exists.")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"\nLoaded existing DB with {vectorstore._collection.count()} chunks")
        return vectorstore

    # Otherwise, create new DB
    print("Creating new vector store...\n")
    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\nIngestion completed successfully.")
    return vectorstore


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
