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
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OpenRouter API key")

    user_msg = chat_req.messages[-1].content.strip()
    user_vector = vectorizer.transform([user_msg])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_idx = int(similarities.argmax())
    matched_question = questions[best_idx]
    matched_answer = answers[best_idx]
    similarity_score = similarities[best_idx]

    # Inject context clearly and strongly
    system_prompt = (
        f"You are Zendawa Assistant, a helpful and knowledgeable assistant for the telepharmacy platform Zendawa in Kenya.\n\n"
        f"Use this relevant FAQ information to guide your answer:\n"
        f"Q: {matched_question}\nA: {matched_answer}\n\n"
        f"If the user asks something unrelated, politely redirect them back to Zendawa services."
    )

    # Final message list
    prompt_messages = [
        {"role": "system", "content": system_prompt},
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
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        res.raise_for_status()
        data = res.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, I don't have that information.")
        return {"reply": reply}
    except Exception as e:
        print("❌ Error:", e)
        return {"reply": "Sorry, I could not retrieve a response at the moment. Please try again later."}

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return Path("static/index.html").read_text()
