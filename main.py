from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, requests
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# === Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "qwen/qwen3-1.7b:free"

# === Classes ===
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# === Setup Vector Store ===
chroma_client = Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="zendawa-knowledge")

# === Load + Embed your FAQ ===
def load_zendawa_data():
    if collection.count() > 0:
        return  # Skip if already loaded
    with open("documents/zendawa_faq.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = text.split("\n\n")
    docs = [chunk.strip() for chunk in chunks if chunk.strip()]
    collection.add(
        documents=docs,
        ids=[f"id-{i}" for i in range(len(docs))]
    )

load_zendawa_data()

# === Main Chat Endpoint ===
@app.post("/chat")
async def chat(chat_req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="API key missing")

    user_msg = chat_req.messages[-1].content
    results = collection.query(query_texts=[user_msg], n_results=3)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    prompt_messages = [
        {"role": "system", "content": f"You are a Zendawa telepharmacy assistant. Use the following knowledge:\n\n{context}"},
    ] + [msg.dict() for msg in chat_req.messages]

    payload = {
        "model": MODEL,
        "messages": prompt_messages,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return {"reply": data["choices"][0]["message"]["content"]}
    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=502, detail="Failed to get OpenRouter response.")
