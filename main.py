from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, requests

# === FastAPI Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Config ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "qwen/qwen3-1.7b:free"

# === Data Model ===
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# === Load FAQ Text File ===
faq_path = "documents/zendawa_faq.txt"

with open(faq_path, "r", encoding="utf-8") as f:
    raw_faq_blocks = f.read().strip().split("\n\n")

# Extract Q&A pairs
faq_pairs = []
for block in raw_faq_blocks:
    lines = block.strip().split("\n")
    q = next((line[3:] for line in lines if line.lower().startswith("q:")), None)
    a = next((line[3:] for line in lines if line.lower().startswith("a:")), None)
    if q and a:
        faq_pairs.append((q.strip(), a.strip()))

# TF-IDF vectorization
questions = [q for q, a in faq_pairs]
answers = [a for q, a in faq_pairs]
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

# === Main Chat Endpoint ===
@app.post("/chat")
async def chat(chat_req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OpenRouter API key")

    user_msg = chat_req.messages[-1].content.strip()
    user_vector = vectorizer.transform([user_msg])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_idx = int(similarities.argmax())
    best_match = questions[best_idx]
    matched_answer = answers[best_idx]
    similarity_score = similarities[best_idx]

    # Construct context-enhanced prompt
    context = f"Relevant Zendawa info:\nQ: {best_match}\nA: {matched_answer}"
    prompt_messages = [
        {"role": "system", "content": context},
        *[msg.dict() for msg in chat_req.messages]
    ]

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
        print("❌ Error:", e)
        raise HTTPException(status_code=502, detail="OpenRouter response failed.")

from fastapi.responses import FileResponse

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
