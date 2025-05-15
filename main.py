from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/mixtral-8x7b-instruct:free"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post("/chat")
async def chat(chat_req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="API key missing")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [msg.dict() for msg in chat_req.messages],
        "stream": False
    }

  try:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers
    )
    response.raise_for_status()  # Raises an HTTPError for bad status codes (e.g., 401, 500)

    data = response.json()
    print("✅ OPENROUTER SUCCESS:", data)  # Log entire response
    return {"reply": data["choices"][0]["message"]["content"]}

except requests.exceptions.HTTPError as http_err:
    print("❌ HTTP Error:", response.status_code)
    print("❌ Error Response:", response.text)
    raise HTTPException(status_code=502, detail=f"HTTP error: {http_err}")

except requests.exceptions.RequestException as req_err:
    print("❌ Request Error:", req_err)
    raise HTTPException(status_code=502, detail=f"Request error: {req_err}")

except Exception as e:
    print("❌ Unexpected Error:", str(e))
    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")


