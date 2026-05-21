# =========================================================
# IMPORT LIBRARIES
# =========================================================

# Semantic chunker for meaning-based text splitting
from langchain_experimental.text_splitter import SemanticChunker

# HuggingFace embeddings for semantic similarity
from langchain_huggingface import HuggingFaceEmbeddings

# Environment variable loader
from dotenv import load_dotenv


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# SAMPLE TEXT
# =========================================================
#
# This text covers:
# 1. Tesla financial results
# 2. Model Y performance
# 3. Production challenges
#
# Semantic chunking groups sentences
# by meaning rather than character count.

tesla_text = """
Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance

The Model Y became the best-selling vehicle globally,
with 350,000 units sold.

Customer satisfaction ratings reached an all-time high of 96%.

Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

Tesla is working to diversify its supplier base.

New manufacturing techniques are being implemented to reduce costs.
"""


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
#
# Semantic chunking relies on embeddings
# to measure sentence similarity.
#
# Sentences with similar meaning
# stay in the same chunk.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# CREATE SEMANTIC CHUNKER
# =========================================================
#
# Unlike RecursiveCharacterTextSplitter:
# - SemanticChunker splits by topic shifts
# - Not by fixed character length
#
# Example:
# - Revenue sentences form one chunk
# - Production issues form another

semantic_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=70           # higher = fewer chunks
)


# =========================================================
# SPLIT TEXT SEMANTICALLY
# =========================================================

chunks = semantic_splitter.split_text(tesla_text)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\nSEMANTIC CHUNKING RESULTS")
print("=" * 60)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}")
    print(f"Chunk Length: {len(chunk)} characters")
    print("\nChunk Content:\n")
    print(chunk)
    print("\n" + "=" * 60)

# =========================================================
# KEY DIFFERENCE
# =========================================================
#
# NORMAL CHUNKING:
# - Splits by character count, sentence count, or separators
#
# SEMANTIC CHUNKING:
# - Splits by meaning similarity and topic boundaries
#
# BENEFIT:
# - Higher retrieval quality in RAG systems
#
# DRAWBACK:
# - Slower ingestion (embeddings computed during chunking)
#
# NOTE:
# - Semantic chunking affects document splitting only
# - Answer generation still uses your gemma:2b model
