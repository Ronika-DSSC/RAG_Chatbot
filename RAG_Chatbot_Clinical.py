import os
import sys
import json
import time
import uuid
import requests

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import chromadb
import gradio as gr
import PyPDF2
import pdfplumber

from docx import Document


# ================================================================= CONFIG ===============================================================
DATA_DIR = "knowledgeBase"
# Number of document chunks to retrieve
TOP_K = 20
# Similarity threshold for documents
COSINE_THRESHOLD = 0.5
# Chat memory retention
CHAT_MEMORY_DAYS = 28
# Chroma persistence directory
CHROMA_DIR = "./chroma_db"
# Toggle backend
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"


# =============================================================== MODELS =================================================================
OLLAMA_LLM = "openai/gpt-oss:120b"
OLLAMA_EMBED = "nomic-ai/nomic-embed-text-v1.5"

# IMPORTANT: Do not hard-code API keys in source code.
LLM_API_KEY = os.environ["LLM_API_KEY"]
EMBEDDING_API_KEY = os.environ["EMBEDDING_API_KEY"]

LLM_URL = "https://llm.research.cchmc.org/api/chat/completions"
EMBEDDING_URL = "https://llm.research.cchmc.org/ollama/api/embed"


# ============================================================ SYSTEM PROMPT =============================================================
SYSTEM_PROMPT = """You are a supportive conversational assistant for patients recovering from concussion.
Your role is to help patients reflect on their recovery, symptoms, activities, feelings, daily experiences, and progress.
Use the patient's current message and relevant conversation memory to respond naturally and supportively.

When clinical information is needed, use only the provided concussion document context. 
Do not introduce medical information that is not supported by that context.

Do not invent or assume symptoms, feelings, experiences, activities, or personal details that the patient has not shared.
For reflection questions, encourage the patient to describe their own experience rather than answering the question for them.

If the available information is insufficient to answer a clinical question,
say that the information is not available in the provided context and encourage
the patient to speak with their medical team.

Do not provide medical advice.
If a question is ambiguous, ask for clarification."""


# ============================================================ CHROMADB ===================================================================
print("Initializing ChromaDB...")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Collection containing concussion document chunks
documents_collection = chroma_client.get_or_create_collection(name="concussion_documents", metadata={ "description": "Concussion clinical document chunks"})
# Collection containing conversational memory
chat_collection = chroma_client.get_or_create_collection(name="chat_memory", metadata={"description": "Persistent conversational memory"})


# ============================================================ EMBEDDINGS =================================================================
# Function to get embeddings
def get_embeddings(texts: List[str], embedding_model: str, embedding_api_key: str) -> List[List[float]]:
    tries = 0
    while tries < 3:
        tries += 1
        try:
            response = requests.post(
                url=EMBEDDING_URL,
                headers={"Authorization": f"Bearer {embedding_api_key}", "Content-Type": "application/json"},
                json={"model": embedding_model, "input": texts, "keep_alive": 0},
                timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            if tries < 3:
                print(f"Embedding error: {e}. " f"Retrying ({tries}/3)...")
                time.sleep(min(tries, 10))
            else:
                raise RuntimeError(f"Embedding service failed after 3 attempts: {e}")


# ========================================================= DOCUMENT READING ==============================================================
# Function to read document from file
def read_docx_file(filename):
    try:
        document = Document(filename)
        full_text = []
        for paragraph in document.paragraphs:
            full_text.append(paragraph.text)
        return "\n".join(full_text)
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"Error reading DOCX {filename}: {e}")
        return ""


def format_markdown_table(table: List[List[Optional[str]]]) -> str:
    if not table:
        return "(empty table)\n"
    rows = [[cell if cell is not None else "" for cell in row] for row in table]
    header = rows[0]
    body = rows[1:]
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        # Make sure row has same number of columns
        row = row + [""] * (len(header) - len(row))
        md.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(md) + "\n"


def extract_text_from_pdf_tables(file_path: str):
    reader = PyPDF2.PdfReader(file_path)
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_pages.append(page_text)
    tables_pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            tables_pages.append(tables)
    combined_output = []
    for i, text in enumerate(text_pages):
        combined_output.append((f"page_{i + 1}", text))
        page_tables = (tables_pages[i] if i < len(tables_pages) else [])
        for t_idx, table in enumerate(page_tables):
            combined_output.append((f"page_{i + 1}, table_{t_idx}", format_markdown_table(table)))
    return combined_output


def extract_text_from_pdf(file_path: str):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_file(file_path):
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension == ".docx":
        return read_docx_file(file_path)
    elif extension == ".pdf":
        pdf_data = extract_text_from_pdf_tables(file_path)
        # Convert page/table tuples into text
        return "\n\n".join(f"{location}\n{text}" for location, text in pdf_data)
    elif extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ============================================================ LOAD DOCUMENTS ============================================================
def load_files(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                text = read_file(path)
                if text:
                    docs.append((file, text))
            except Exception as e:
                print(f"Could not read {path}: {e}")
    return docs


# ============================================================ CHUNK DOCUMENTS ============================================================
def chunk_docs(docs, chunk_size=300, overlap=50):
    chunks = []
    step = chunk_size - overlap
    for source, text in docs:
        words = text.split()
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + chunk_size])
            if not chunk_text.strip():
                continue
            chunks.append({"text": chunk_text, "source": source, "chunk_id": len(chunks)})
    return chunks


# ==================================================== BUILD CHROMADB DOCUMENT COLLECTION =================================================
def build_chroma_document_index(chunks_data):
    if not chunks_data:
        print("No document chunks found.")
        return

    # IMPORTANT: Don't recreate embeddings every time the application starts if Chroma already contains the documents.
    existing_count = documents_collection.count()
    if existing_count > 0:
        print(f"ChromaDB already contains " f"{existing_count} document chunks.")
        return
    print(f"Creating embeddings for " f"{len(chunks_data)} document chunks...")
    embedding_batch_size = 25
    embeddings = []
    texts = [chunk["text"] for chunk in chunks_data]
    for start in range(0, len(texts), embedding_batch_size):
        end = min(start + embedding_batch_size, len(texts))
        print(f"Embedding chunks {start} - {end} " f"of {len(texts)}...")
        batch_embeddings = get_embeddings(texts[start:end], OLLAMA_EMBED, EMBEDDING_API_KEY)
        embeddings.extend(batch_embeddings)
        print(f"Received {len(batch_embeddings)} embeddings.")

    if len(embeddings) != len(chunks_data):
        raise RuntimeError(f"Expected {len(chunks_data)} embeddings, " f"but received {len(embeddings)}.")

    ids, documents, metadatas = [],[],[];
    for i, chunk in enumerate(chunks_data):
        ids.append(f"document_chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"], "chunk_id": i})

    batch_size = 50
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        print(f"Adding document chunks " f"{start} - {end}")
        documents_collection.add(ids=ids[start:end], documents=documents[start:end], embeddings=embeddings[start:end], metadatas=metadatas[start:end])
    print(f"Added {len(ids)} document chunks to ChromaDB.")

# ============================================================ RETRIEVE DOCUMENTS ============================================================
def retrieve(query, k=TOP_K):
    query_embedding = get_embeddings([query], OLLAMA_EMBED, EMBEDDING_API_KEY)[0]
    results = documents_collection.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])
    retrieved = []
    if not results["documents"]:
        return retrieved

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        # Chroma distance depends on collection metric. # With cosine distance: similarity = 1 - distance
        similarity = 1 - distance
        if similarity >= COSINE_THRESHOLD:
            retrieved.append({"text": document, "source": metadata["source"], "score": similarity})
    return retrieved


# =============================================================== CHAT MEMORY ================================================================
def save_chat_memory(user_id, conversation_id, user_message, assistant_message):
    timestamp = datetime.now(timezone.utc).isoformat()
    user_message_id = str(uuid.uuid4())
    assistant_id = str(uuid.uuid4())
    # Embed both messages
    embeddings = get_embeddings([user_message, assistant_message], OLLAMA_EMBED, EMBEDDING_API_KEY)
    chat_collection.add(
        ids=[user_message_id, assistant_id], 
        documents=[user_message, assistant_message], 
        embeddings=embeddings,
        metadatas=[{"user_id": user_id, "conversation_id": conversation_id, "role": "user", "timestamp": timestamp},
            {"user_id": user_id, "conversation_id": conversation_id, "role": "assistant", "timestamp": timestamp}])


def get_recent_chat_memory(user_id):
    cutoff = datetime.now(timezone.utc) - timedelta(days=CHAT_MEMORY_DAYS)
    # Only filter by user_id in ChromaDB.
    results = chat_collection.get(where={"user_id": user_id}, include=["documents", "metadatas"])
    memories = []
    for document, metadata in zip(results["documents"], results["metadatas"]):
        timestamp = datetime.fromisoformat(metadata["timestamp"])
        # Enforce 28-day retention in Python.
        if timestamp < cutoff:
            continue
        memories.append({"role": metadata["role"], "content": document, "timestamp": metadata["timestamp"], "conversation_id": metadata["conversation_id"]})
    memories.sort(key=lambda x: x["timestamp"])
    return memories


# =========================================================== SEMANTIC CHAT MEMORY ============================================================
def retrieve_relevant_chat_memory(query, user_id, k=10):
    query_embedding = get_embeddings([query], OLLAMA_EMBED, EMBEDDING_API_KEY)[0]
    # Restrict memory retrieval to this user.
    results = chat_collection.query(query_embeddings=[query_embedding], n_results=k, where={"user_id": user_id}, include=["documents", "metadatas", "distances"])

    memories = []
    if not results["documents"]:
        return memories

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    cutoff = (datetime.now(timezone.utc) - timedelta(days=CHAT_MEMORY_DAYS))
    for document, metadata, distance in zip(documents, metadatas, distances):
        timestamp = datetime.fromisoformat(metadata["timestamp"])
        # Enforce 28-day retention
        if timestamp < cutoff:
            continue
        memories.append({"role": metadata["role"], "content": document, "timestamp": metadata["timestamp"], "distance": distance})
    return memories


# ========================================================== CLEANUP OLD CHAT MEMORY ===========================================================
def cleanup_old_chat_memory():
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CHAT_MEMORY_DAYS))
    results = chat_collection.get(include=["metadatas"])
    ids_to_delete = []
    for item_id, metadata in zip(results["ids"], results["metadatas"]):
        timestamp_string = metadata.get("timestamp")
        if not timestamp_string:
            continue
        try:
            timestamp = datetime.fromisoformat(timestamp_string)
            if timestamp < cutoff:
                ids_to_delete.append(item_id)
        except ValueError:
            continue
    if ids_to_delete:
        chat_collection.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} " f"chat records older than " f"{CHAT_MEMORY_DAYS} days.")


# ============================================================== GENERATE ANSWER ================================================================
def generate_answer(query, context_chunks, user_id, system_prompt, temperature):
    # DOCUMENT CONTEXT
    context = "\n\n".join([(f"[Source: {c['source']}]\n" f"{c['text']}") for c in context_chunks])
    # RECENT CONVERSATION
    recent_memory = get_recent_chat_memory(user_id)
    # RELEVANT OLDER CONVERSATION
    relevant_memory = retrieve_relevant_chat_memory(query, user_id, k=10)
    # Remove duplicates
    seen = set()
    combined_memory = []
    for memory in (recent_memory + relevant_memory):
        key = (memory["role"], memory["content"])
        if key not in seen:
            seen.add(key)
            combined_memory.append(memory)
    # Sort chronologically
    combined_memory.sort(key=lambda x: x["timestamp"])
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    # Conversation memory
    if combined_memory:
        memory_text = "\n\n".join([(f"{m['role'].upper()}: " f"{m['content']}") for m in combined_memory])
        messages.append({"role": "system", "content": "CONVERSATION MEMORY " "(from the last 28 days):\n\n" + memory_text})
    # Current document context
    prompt = f"""CONTEXT FROM CONCUSSION DOCUMENTS: \n{context}\n\nCURRENT QUESTION: \n{query}"""
    messages.append({"role": "user", "content": prompt})

    # ------------------------------------ Call LLM -----------------------------------
    response = requests.post(url=LLM_URL,
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
        json={"model": OLLAMA_LLM, "temperature": temperature, "messages": messages, "keep_alive": 0, "stream": False},
        timeout=300)

    response.raise_for_status()
    data = response.json()
    answer = (data["choices"][0]["message"]["content"])
    sources = list(set(c["source"]for c in context_chunks))
    return answer, sources


# ============================================================== CHAT FUNCTION ==============================================================
def chat_fn(message, user_id, conversation_id, system_prompt, temperature):
    # Retrieve documents
    retrieved = retrieve(message)
    # Generate answer
    answer, sources = generate_answer(message, retrieved, user_id, system_prompt,temperature)
    # Save conversation
    save_chat_memory(user_id, conversation_id, message, answer)
    # Return answer
    if sources:
        return (f"{answer}\n\n" "Sources:\n" + "\n".join(sources))
    return answer


# ================================================================ RESET CHAT=================================================================
def reset_chat():
    # New conversation ID means the new chat won't see the previous conversation.
    return ([], "", str(uuid.uuid4()))


def switch_user(user_id):
    """
    Switch to a different test user and start a new conversation. The user's previous conversations remain stored in ChromaDB.
    """
    new_conversation_id = str(uuid.uuid4())
    return ([], "", new_conversation_id)


# ================================================================ INITIALIZE ================================================================
print("Loading documents...")
docs = load_files(DATA_DIR)
print(f"Loaded {len(docs)} documents.")

print("Chunking...")
chunks_data = chunk_docs(docs)
print(f"Total chunks: {len(chunks_data)}")

print("Building ChromaDB document index...")
build_chroma_document_index(chunks_data)

print("Cleaning old chat memory...")
cleanup_old_chat_memory()


# =================================================================== GRADIO =================================================================
print("Starting Gradio...")
with gr.Blocks() as demo:
    gr.Markdown("## Concussion RAG Chatbot")
    # User IDs
    user_selector = gr.Dropdown(choices=["test_user_1", "test_user_2", "test_user_3"], value="test_user_1", label="Test User")
    # Each browser/session gets its own conversation ID
    conversation_id = gr.State(str(uuid.uuid4()))
    system_prompt_box = gr.Textbox(label="System Prompt", value=SYSTEM_PROMPT, lines=5)
    temperature_slider = gr.Slider(minimum=0.0, maximum=1.5, value=0.5, step=0.1, label="Temperature")

    with gr.Row():
        msg = gr.Textbox(label="Your question", scale=4)
        new_chat_btn = gr.Button("New Chat", scale=1)
    with gr.Row():
        chatbot = gr.Chatbot(height=600)

    def respond(message, chat_history, user_id, conversation_id, system_prompt, temperature):
        reply = chat_fn(message, user_id, conversation_id, system_prompt, temperature)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return ("", chat_history)

    # New chat
    new_chat_btn.click(fn=reset_chat, inputs=[], outputs=[chatbot, msg, conversation_id])
    # Switch user
    user_selector.change(fn=switch_user, inputs=[user_selector], outputs=[chatbot, msg, conversation_id])
    # Submit
    msg.submit(fn=respond, inputs=[msg, chatbot, user_selector, conversation_id, system_prompt_box, temperature_slider], outputs=[msg, chatbot])


# ================================================================== RUN =====================================================================
if __name__ == "__main__":
    print("HERE")
    demo.launch()
