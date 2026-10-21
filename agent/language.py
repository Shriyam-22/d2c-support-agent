"""Language detection and Hinglish normalisation.

Indian D2C customers write in three main registers:
  - English
  - Hindi in Devanagari script
  - Hinglish: Hindi written in Roman script, often mixed with English
    ("mera order kab aayega?")

We detect the register so the agent can reply in the same one, and we
normalise Hinglish words to English before retrieval, because the catalogue
and policies are written in English.
"""
from __future__ import annotations

import re

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
TOKEN = re.compile(r"[a-z0-9]+")

# Common Roman-script Hindi function words. Two or more of these in a message
# is a strong Hinglish signal; English rarely contains them.
HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kyu", "kyun", "mera", "meri", "mere", "mujhe", "kab",
    "kaise", "kitna", "kitne", "nahi", "nahin", "chahiye", "aap", "aapka", "bhai",
    "kar", "karo", "karna", "raha", "rahi", "tha", "thi", "ho", "hoga", "gaya",
    "gayi", "abhi", "kuch", "koi", "accha", "acha", "batao", "bataiye", "wala",
    "wali", "liye", "aayega", "aaya", "milega", "mila", "paisa", "paise", "yaar",
    "hua", "kal", "aaj", "kahan", "lekin", "par", "se", "ka", "ki", "ke",
}

# Hinglish/Hindi word -> English term used in the catalogue or policies.
# Used only for retrieval, never shown to the customer.
HINGLISH_TO_ENGLISH = {
    "kab": "when delivery", "aayega": "delivery", "aaya": "delivered",
    "pahuncha": "delivered", "kahan": "where track", "wapas": "return",
    "wapsi": "return refund", "lautana": "return", "badalna": "exchange",
    "paisa": "refund", "paise": "refund", "kitna": "price", "kitne": "price",
    "daam": "price", "dam": "price", "sasta": "cheap", "mehenga": "expensive",
    "baal": "hair", "jhadna": "hair fall", "jhad": "hair fall", "rusi": "dandruff",
    "chehra": "face", "chehre": "face", "twacha": "skin", "tvacha": "skin",
    "rukhi": "dry skin", "sukhi": "dry skin", "chikni": "oily skin",
    "tel": "oil", "daane": "acne pimples", "muhase": "acne pimples",
    "dhoop": "sunscreen sun", "garmi": "summer", "sardi": "winter",
    "toota": "damaged", "tuta": "damaged", "galat": "wrong item",
    "radd": "cancel", "cancel": "cancel", "pata": "address", "naam": "name",
}

HINDI_DEVANAGARI_TO_ENGLISH = {
    "ऑर्डर": "order", "आर्डर": "order", "कब": "when delivery", "आएगा": "delivery",
    "वापस": "return", "रिफंड": "refund", "पैसे": "refund", "कीमत": "price",
    "बाल": "hair", "रूसी": "dandruff", "चेहरा": "face", "त्वचा": "skin",
    "रद्द": "cancel", "टूटा": "damaged",
}


def detect_language(text: str) -> str:
    """Return 'hi' (Devanagari), 'hinglish' (Roman-script Hindi) or 'en'."""
    if not text:
        return "en"
    if DEVANAGARI.search(text):
        return "hi"
    tokens = TOKEN.findall(text.lower())
    if not tokens:
        return "en"
    hits = sum(1 for t in tokens if t in HINGLISH_MARKERS)
    # Short messages: 2 markers is enough. Longer ones: require ~15% density.
    if hits >= 2 and (len(tokens) <= 12 or hits / len(tokens) >= 0.15):
        return "hinglish"
    return "en"


def normalise_for_search(text: str) -> str:
    """Append English equivalents of Hinglish/Hindi words to help retrieval.

    The original text is kept, so English words still match as usual.
    """
    extra: list[str] = []
    lowered = text.lower()
    for token in TOKEN.findall(lowered):
        if token in HINGLISH_TO_ENGLISH:
            extra.append(HINGLISH_TO_ENGLISH[token])
    for word, english in HINDI_DEVANAGARI_TO_ENGLISH.items():
        if word in text:
            extra.append(english)
    return f"{text} {' '.join(extra)}".strip() if extra else text


LANGUAGE_INSTRUCTIONS = {
    "en": "The customer is writing in English. Reply in simple, friendly English.",
    "hi": "The customer is writing in Hindi (Devanagari). Reply in simple Hindi using Devanagari script. Keep product names, order IDs and prices as they are.",
    "hinglish": "The customer is writing in Hinglish (Hindi in Roman script). Reply in natural Hinglish in Roman script, the way a friendly Indian support agent would text on WhatsApp. Keep product names, order IDs and prices as they are.",
}
