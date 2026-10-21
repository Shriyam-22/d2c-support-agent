"""WhatsApp Cloud API helpers.

Parsing is pure Python and unit-tested. Sending needs a Meta developer app,
a WhatsApp Business phone number ID and an access token; until those are set,
replies are just returned by the webhook so you can test locally.
Check Meta's current WhatsApp Cloud API docs before going live: the Graph API
version and payload details change over time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class IncomingMessage:
    from_number: str
    text: str
    message_id: str
    name: str | None = None


def parse_webhook(payload: dict) -> list[IncomingMessage]:
    """Extract text messages from a WhatsApp Cloud API webhook payload.
    Non-text messages (images, status updates, reactions) are ignored."""
    messages = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            names = {c.get("wa_id"): c.get("profile", {}).get("name") for c in value.get("contacts", [])}
            for msg in value.get("messages", []):
                if msg.get("type") != "text":
                    continue
                sender = msg.get("from", "")
                messages.append(IncomingMessage(
                    from_number=sender,
                    text=msg.get("text", {}).get("body", ""),
                    message_id=msg.get("id", ""),
                    name=names.get(sender),
                ))
    return messages


def send_text(to_number: str, text: str) -> bool:
    """Send a reply through the Cloud API. Returns False if not configured."""
    token = os.getenv("WHATSAPP_TOKEN")
    phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    version = os.getenv("WHATSAPP_API_VERSION", "v21.0")
    if not token or not phone_id:
        return False
    import requests

    resp = requests.post(
        f"https://graph.facebook.com/{version}/{phone_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        json={"messaging_product": "whatsapp", "to": to_number, "type": "text", "text": {"body": text}},
        timeout=10,
    )
    return resp.ok
