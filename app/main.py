"""HTTP API.

  POST /chat               simple JSON endpoint for local testing and demos
  GET  /webhook/whatsapp   Meta webhook verification handshake
  POST /webhook/whatsapp   incoming WhatsApp messages

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import os
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from agent.core import build_agent, trim_history

from .whatsapp import parse_webhook, send_text

app = FastAPI(title="Neem & Nest support agent")
agent = build_agent()

# In-memory sessions keyed by phone number. Use Redis or a database in
# production so history survives restarts and works across instances.
SESSIONS: dict[str, list[dict]] = defaultdict(list)
SEEN_MESSAGE_IDS: set[str] = set()  # WhatsApp can deliver the same webhook twice


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    tools_called: list[str]
    flags: list[str]
    latency_ms: float
    cost_usd: float


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = agent.respond(req.message, trim_history(SESSIONS[req.session_id]), session_id=req.session_id)
    SESSIONS[req.session_id] = result.messages
    t = result.trace
    return ChatResponse(reply=result.reply, tools_called=t.tools_called, flags=result.flags,
                        latency_ms=round(t.total_latency_ms, 1), cost_usd=round(t.cost_usd, 6))


@app.get("/webhook/whatsapp", response_class=PlainTextResponse)
def verify(mode: str = Query(alias="hub.mode", default=""),
           token: str = Query(alias="hub.verify_token", default=""),
           challenge: str = Query(alias="hub.challenge", default="")) -> str:
    if mode == "subscribe" and token and token == os.getenv("WHATSAPP_VERIFY_TOKEN"):
        return challenge
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook/whatsapp")
async def incoming(request: Request) -> dict:
    payload = await request.json()
    replies = []
    for msg in parse_webhook(payload):
        if msg.message_id in SEEN_MESSAGE_IDS:
            continue
        SEEN_MESSAGE_IDS.add(msg.message_id)
        result = agent.respond(msg.text, trim_history(SESSIONS[msg.from_number]), session_id=msg.from_number)
        SESSIONS[msg.from_number] = result.messages
        sent = send_text(msg.from_number, result.reply)
        replies.append({"to": msg.from_number, "reply": result.reply, "sent": sent})
    # Always return 200 quickly, or WhatsApp will retry the webhook.
    return {"status": "ok", "replies": replies}
