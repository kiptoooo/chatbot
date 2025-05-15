from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import os, requests

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Model Setup ===
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# === Load FAQ Data ===
faq_path = "documents/zendawa_faq.txt"
with open(faq_path, "r", encoding="utf-8") as f:
    raw_faq_blocks = f.read().strip().split("\n\n")

faq_pairs = []
for block in raw_faq_blocks:
    lines = block.strip().split("\n")
    q = next((line[3:] for line in lines if line.lower().startswith("q:")), None)
    a = next((line[3:] for line in lines if line.lower().startswith("a:")), None)
    if q and a:
        faq_pairs.append((q.strip(), a.strip()))

questions = [q for q, a in faq_pairs]
answers = [a for q, a in faq_pairs]
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

@app.post("/chat")
async def chat(chat_req: ChatRequest):
    if not TOGETHER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Together API key")

    user_msg = chat_req.messages[-1].content.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Empty user message")

    user_vector = vectorizer.transform([user_msg])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = int(similarities.argmax())
    best_match = questions[best_idx]
    matched_answer = answers[best_idx]

    context = f"Relevant Zendawa info:\nQ: {best_match}\nA: {matched_answer}"
    prompt_messages = [
        {"role": "system", "content": context},
        *[{"role": m.role, "content": m.content} for m in chat_req.messages]
    ]

    payload = {
        "model": MODEL,
        "messages": prompt_messages,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "choices" in data:
            return {"reply": data["choices"][0]["message"]["content"]}
        else:
            raise HTTPException(status_code=502, detail="Invalid response format from Together.ai")
    except Exception as e:
        print("❌ Error:", e)
        raise HTTPException(status_code=502, detail="Together.ai request failed.")

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return Path("static/index.html").read_text()
