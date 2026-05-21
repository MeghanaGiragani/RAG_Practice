# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# ChromaDB vector store
from langchain_chroma import Chroma

# HuggingFace embedding model (free, lightweight)
from langchain_huggingface import HuggingFaceEmbeddings

# Optional: load environment variables (not used here)
# from dotenv import load_dotenv
# load_dotenv()


# =========================================================
# PATH TO CHROMADB
# =========================================================
#
# In ingestion:
# - text chunks + embeddings
# were stored inside "db/chroma_db".
#
# Now we reload the same database.

persistent_directory = "db/chroma_db"


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
#
# Critical: use the SAME embedding model
# as ingestion. Otherwise, query vectors
# won’t align with stored document vectors.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD EXISTING VECTOR STORE
# =========================================================
#
# Loads chunks, embeddings, and metadata
# from disk into memory.

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  # cosine similarity
)


# =========================================================
# USER QUERY
# =========================================================

query = "How much did Microsoft pay to acquire GitHub?"


# =========================================================
# CREATE RETRIEVER
# =========================================================
#
# Retriever workflow:
# 1. Convert query → embedding vector
# 2. Compare with stored vectors
# 3. Return top-k most similar chunks

retriever = db.as_retriever(search_kwargs={"k": 5})


# =========================================================
# OPTIONAL ADVANCED RETRIEVER
# =========================================================
#
# similarity_score_threshold:
# Only return chunks above a minimum similarity.
# Uncomment if stricter filtering is needed.

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 5, "score_threshold": 0.3}
# )


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================
#
# Flow:
# Query → vector → similarity search → nearest chunks

relevant_docs = retriever.invoke(query)

# =========================================================
# DISPLAY RESULTS
# =========================================================

print(f"\nUser Query: {query}")
print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\n========== Document {i} ==========")
    print(doc.page_content)       # chunk content
    print("\n--- Metadata ---")
    print(doc.metadata)           # metadata
    print("\n" + "=" * 50)


# =========================================================
# INTERNAL PROCESS (EXAMPLE)
# =========================================================
#
# Query: "How much did Microsoft pay to acquire GitHub?"
#
# Step 1: Query → embedding vector
# Step 2: Compare with all stored vectors
# Step 3: Find closest matches
# Step 4: Return relevant chunks
#
# Example retrieved chunk:
# "Microsoft acquired GitHub in 2018 for $7.5 billion..."
#
# In a full RAG pipeline:
# retrieved chunks + user query → LLM
# → final answer generation.
#
# Current stage: Retrieval only
# Next stage: Retrieval + Generation
