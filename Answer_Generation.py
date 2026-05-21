# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# ChromaDB vector database
# Used to store and retrieve embeddings
from langchain_chroma import Chroma

# Free HuggingFace embedding model
# Converts text into vectors
from langchain_huggingface import HuggingFaceEmbeddings

# OpenAI chat model
# Used for final answer generation
from langchain_ollama import ChatOllama

# Message types for LLM conversation
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

# Loads .env variables
# from dotenv import load_dotenv


# Load environment variables
#
# Example:
#
# OPENAI_API_KEY=xxxxxxxx
#
# from .env file
# load_dotenv()


# =========================================================
# VECTOR DATABASE LOCATION
# =========================================================
#
# This folder was created during ingestion pipeline.
#
# It contains:
#
# - chunks
# - embeddings
# - metadata

persistent_directory = "db/chroma_db"


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
#
# IMPORTANT:
#
# Use SAME embedding model
# used during ingestion.
#
# During ingestion:
#
# text -> vectors
#
# During retrieval:
#
# query -> vectors
#
# If different embedding models are used,
# retrieval quality becomes bad.

embedding_model = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD CHROMADB VECTOR STORE
# =========================================================
#
# Loads stored embeddings + chunks.

db = Chroma(

    # Folder path
    persist_directory=persistent_directory,

    # Embedding model used for query embedding
    embedding_function=embedding_model,

    # Similarity metric
    collection_metadata={
        "hnsw:space": "cosine"
    }
)


# =========================================================
# USER QUESTION
# =========================================================

query = "Founder of spaceX"


# =========================================================
# CREATE RETRIEVER
# =========================================================
#
# Retriever does:
#
# Query
#   ↓
# Convert query into vector
#   ↓
# Compare with stored vectors
#   ↓
# Return nearest chunks

retriever = db.as_retriever(

    search_kwargs={

        # Return top 5 chunks
        "k": 10
    }
)


# =========================================================
# OPTIONAL ADVANCED RETRIEVAL
# =========================================================
#
# Uncomment if needed.
#
# Returns only chunks above
# certain similarity score.

# retriever = db.as_retriever(
#
#     search_type="similarity_score_threshold",
#
#     search_kwargs={
#
#         "k": 5,
#
#         # Minimum similarity score
#         "score_threshold": 0.3
#     }
# )


# =========================================================
# RETRIEVE RELEVANT DOCUMENTS
# =========================================================
#
# Example flow:
#
# User Query:
#
# "Microsoft GitHub acquisition"
#
# Query embedding generated
# ↓
# Compared with all chunk embeddings
# ↓
# Similar chunks retrieved

relevant_docs = retriever.invoke(query)


# =========================================================
# DISPLAY RETRIEVED CONTEXT
# =========================================================

print(f"\nUser Query: {query}")

print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):

    print(f"\n========== Document {i} ==========")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("\n" + "=" * 50)


# =========================================================
# COMBINE CONTEXT + USER QUESTION
# =========================================================
#
# This is the MOST IMPORTANT part in RAG.
#
# We combine:
#
# 1. User query
# 2. Retrieved chunks
#
# and send them to LLM.
#
# This grounding process helps reduce hallucination.

combined_input = f"""
Based on the following documents,
please answer this question:

Question:
{query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Instructions:
- Answer ONLY using the provided documents.
- If answer is not found,
  say:
  "I don't have enough information."
"""


# =========================================================
# CREATE CHAT MODEL
# =========================================================
#
# GPT-4o is used ONLY for answer generation.
#
# Embeddings and generation are separate.
#
# Embeddings:
# HuggingFace (free)
#
# Generation:
# OpenAI GPT-4o (paid)

model = ChatOllama(model="gemma:2b")


# =========================================================
# CREATE MESSAGE LIST
# =========================================================
#
# SystemMessage:
# controls AI behavior
#
# HumanMessage:
# actual user input/context

messages = [

    # System instructions
    SystemMessage(
        content=
        "You are a helpful assistant."
    ),

    # User question + retrieved context
    HumanMessage(
        content=combined_input
    ),
]


# =========================================================
# GENERATE FINAL ANSWER
# =========================================================
#
# LLM receives:
#
# - user query
# - retrieved chunks
#
# and generates grounded answer.

result = model.invoke(messages)


# =========================================================
# DISPLAY FINAL RESPONSE
# =========================================================

print("\n--- Generated Response ---")

print(result.content)


# =========================================================
# COMPLETE RAG FLOW
# =========================================================
#
# STEP 1:
# Load documents
#
# STEP 2:
# Split into chunks
#
# STEP 3:
# Convert chunks into embeddings
#
# STEP 4:
# Store embeddings in ChromaDB
#
# -----------------------------
# QUERY TIME
# -----------------------------
#
# STEP 5:
# User asks question
#
# STEP 6:
# Query converted into embedding
#
# STEP 7:
# Similar chunks retrieved
#
# STEP 8:
# Retrieved chunks + query
# sent to LLM
#
# STEP 9:
# LLM generates final answer
#
#
# THIS IS COMPLETE BASIC RAG.
#
#
# Your Current Architecture:
#
# Documents
#    ↓
# Chunking
#    ↓
# HuggingFace Embeddings (FREE)
#    ↓
# ChromaDB
#    ↓
# Retrieval
#    ↓
# GPT-4o Generation
#    ↓
# Final Answer
#
#
# NEXT LEVEL TOPICS:
#
# - Conversational RAG
# - Memory
# - Hybrid Search
# - Re-ranking
# - Parent Document Retrieval
# - Multi-query Retrieval
# - Agentic RAG
# - Graph RAG
