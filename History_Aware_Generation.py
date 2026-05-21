# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

# Load environment variables
from dotenv import load_dotenv

# Chroma vector database
from langchain_chroma import Chroma

# Message types for conversation
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# HuggingFace embeddings (local, free)
from langchain_huggingface import HuggingFaceEmbeddings

# Ollama local LLM
from langchain_ollama import ChatOllama


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================
#
# This DB already contains:
# - text chunks
# - embeddings
# created during ingestion.

persistent_directory = "db/chroma_db"


# =========================================================
# LOAD SAME EMBEDDING MODEL
# =========================================================
#
# Important: must use the SAME model
# as ingestion, otherwise query vectors
# won’t align with stored vectors.

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)


# =========================================================
# LOAD OLLAMA MODEL
# =========================================================
#
# Local/free LLM. Ensure:
#   ollama pull gemma:2b
# has been run beforehand.

model = ChatOllama(model="gemma:2b")


# =========================================================
# CHAT HISTORY STORAGE
# =========================================================
#
# Keeps track of:
# - user messages
# - AI responses
#
# Enables conversational memory.

chat_history = []


# =========================================================
# MAIN QUESTION FUNCTION
# =========================================================

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # STEP 1 -> HISTORY-AWARE QUESTION REWRITING
    # ------------------------------------------
    # Example:
    # User: "Tell me about Tesla"
    # Next: "Who is the CEO?"
    # → Rewritten: "Who is the CEO of Tesla?"
    # This avoids ambiguity during retrieval.

    if chat_history:
        messages = [
            SystemMessage(content="""
                Rewrite the user's new question
                into a standalone searchable query
                using conversation history.
                Return ONLY the rewritten question.
            """)
        ] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"\nRewritten Search Question:\n{search_question}")
    else:
        search_question = user_question

    # STEP 2 -> RETRIEVE DOCUMENTS
    # -----------------------------
    retriever = db.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(search_question)

    print(f"\nFound {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata['source']}")
        print(doc.page_content[:300])  # preview
        print("-" * 50)

    # STEP 3 -> CREATE FINAL PROMPT
    # ------------------------------
    combined_input = f"""
    Based on the following documents,
    answer the user's question.

    User Question:
    {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Instructions:
    - Answer ONLY from the provided documents.
    - If answer is missing, say:
      "I don't have enough information."
    - Keep answer clear and concise.
    """

    # STEP 4 -> GENERATE FINAL ANSWER
    # --------------------------------
    messages = [
        SystemMessage(content="""
            You are a helpful AI assistant.
            Answer ONLY using the provided documents
            and conversation history.
        """)
    ] + chat_history + [HumanMessage(content=combined_input)]

    result = model.invoke(messages)
    answer = result.content

    # STEP 5 -> UPDATE CHAT HISTORY
    # ------------------------------
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("\n--- Generated Answer ---")
    print(answer)
    return answer


# =========================================================
# SIMPLE CHAT LOOP
# =========================================================

def start_chat():
    print("\n=== HISTORY-AWARE RAG CHATBOT ===")
    print("\nType 'quit' to exit.\n")

    while True:
        question = input("\nYour Question: ")
        if question.lower() == "quit":
            print("\nGoodbye!")
            break
        ask_question(question)


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    start_chat()
