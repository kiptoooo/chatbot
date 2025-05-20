import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = FastAPI()
# Removed HTTPSRedirectMiddleware for local HTTP usage; enable manually if using HTTPS in production.
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === AWS Bedrock / Claude 3 Setup ===
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # Claude 3 Sonnet is available in us-east-1 and us-west-2:contentReference[oaicite:1]{index=1}.
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet model ID on Bedrock:contentReference[oaicite:2]{index=2}.
# Initialize Bedrock runtime client (credentials must be configured via environment or IAM role).
try:
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
except Exception as e:
    raise RuntimeError(f"Error initializing Bedrock client: {e}")

# === Pydantic data models ===
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# === Load FAQ Data (for TF-IDF similarity) ===
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
    # Ensure AWS credentials are available (through env or IAM role)
    if boto3.Session().get_credentials() is None:
        raise HTTPException(status_code=500, detail="Missing AWS credentials for Bedrock API.")
    
    # Get the latest user message content
    user_msg = chat_req.messages[-1].content.strip()
    # Compute similarity between user message and FAQ questions
    user_vector = vectorizer.transform([user_msg])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_idx = int(similarities.argmax())
    matched_question = questions[best_idx]
    matched_answer = answers[best_idx]

    # Construct system prompt with relevant FAQ info and guidance
    system_prompt = (
        "You are Zendawa Assistant, a helpful AI providing accurate, friendly information about Zendawa — "
        "a Kenyan telepharmacy platform offering drug ordering, pharmacy onboarding, teleconsultations, and healthcare logistics.\n\n"
        "If a question falls outside Zendawa’s scope (e.g., about cars, sports, or unrelated topics), respond with a gentle prompt like:\n"
        "\"I'm here to help with questions related to Zendawa’s telepharmacy services. Feel free to ask anything about our platform or healthcare-related support.\"\n\n"
        f"To assist you better, here’s the most relevant information from Zendawa’s FAQ:\nQ: {matched_question}\nA: {matched_answer}"
    )
    # Assemble the conversation context for Claude (system + previous messages)
    bedrock_messages = [
        {"role": "system", "content": [{"text": system_prompt}]}
    ]
    for msg in chat_req.messages:
        # Include each previous message in the conversation (user or assistant)
        bedrock_messages.append({
            "role": msg.role,
            "content": [{"text": msg.content}]
        })

    # Prepare payload for Claude 3 Sonnet (Anthropic Claude message format)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",  # required version identifier for Bedrock Claude API
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9,
        "messages": bedrock_messages
    }

    try:
        # Invoke the Claude model via Bedrock
        response = bedrock_client.invoke_model(modelId=MODEL_ID, body=json.dumps(payload))
        response_body = json.loads(response["body"].read())
        # Extract assistant's reply text from response content
        reply_blocks = [block["text"] for block in response_body.get("content", []) if "text" in block]
        reply_text = "".join(reply_blocks).strip()
        if not reply_text:
            reply_text = "Sorry, I don't have that info."
        return {"reply": reply_text}
    except Exception as e:
        print("❌ Bedrock API Error:", e)
        return {"reply": "Sorry, I could not retrieve a response. Please try again later."}

@app.get("/", response_class=HTMLResponse)
def get_ui():
    # Serve the frontend UI
    return Path("static/index.html").read_text()
