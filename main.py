from fastapi import FastAPI, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os, requests

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)
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
        raise HTTPException(status_code=500, detail="Missing Together.ai API key")

    user_msg = chat_req.messages[-1].content.strip()
    low = user_msg.lower()

    # ——— Handle “Order drugs” intent ————————————————
    if any(kw in low for kw in ["order drug", "buy drug", "order meds", "purchase drug"]):
        return {
            "reply": (
                "🛒 To order drugs on Zendawa:\n"
                "1. Go to our Marketplace: https://zendawa.africa/marketplace\n"
                "2. Browse or search for the medications you need.\n"
                "3. Add your selections to the cart.\n"
                "4. Proceed to checkout and enter delivery details.\n"
                "5. Complete payment and confirm your order.\n\n"
                "👉 Start ordering here: https://zendawa.africa/marketplace"
            )
        }

    # ——— Handle “Consultation” intent ——————————————
    if any(kw in low for kw in ["consultation", "consult", "teleconsult", "book a doctor"]):
        return {
            "reply": (
                "💬 To book a consultation with a doctor:\n"
                "1. Visit: https://zendawa.africa/appointment\n"
                "2. Sign in or create a Zendawa account.\n"
                "3. Select “Teleconsultation” and pick a specialty.\n"
                "4. Choose a date, time, and consultation mode (chat/video).\n"
                "5. Confirm and await your appointment link.\n\n"
                "👉 Book a consultation now: https://zendawa.africa/appointment"
            )
        }

    # ——— Otherwise, your existing FAQ + LLM logic —————————
    # TF-IDF similarity lookup
    user_vector = vectorizer.transform([user_msg])
    sims = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = int(sims.argmax())
    matched_q = questions[best_idx]
    matched_a = answers[best_idx]

    system_prompt = (
        "You are Zendawa Assistant, a helpful AI designed to support users with accurate and friendly information "
        "about Zendawa — a Kenyan telepharmacy platform offering services like drug ordering, pharmacy onboarding, "
        "teleconsultations, and healthcare logistics.\n\n"
        "If a question falls outside Zendawa’s scope (e.g., about cars, sports, or unrelated topics), kindly guide "
        "the user with a gentle message like:\n"
        "“I'm here to help with questions related to Zendawa’s telepharmacy services. Feel free to ask anything "
        "about our platform or healthcare-related support.”\n\n"
        f"To assist you better, here’s the most relevant information from Zendawa’s FAQ:\nQ: {matched_q}\nA: {matched_a}"
    )

    prompt_messages = [{"role": "system", "content": system_prompt}] + [msg.dict() for msg in chat_req.messages]
    payload = {"model": MODEL, "messages": prompt_messages}
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content",
                      "Sorry, I don't have that info.")
        return {"reply": reply}
    except Exception as e:
        print("❌ Error:", e)
        return {"reply": "Sorry, I could not retrieve a response. Please try again later."}

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return Path("static/index.html").read_text()
