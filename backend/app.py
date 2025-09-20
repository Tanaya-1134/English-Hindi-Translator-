# backend/app.py
import os
import tempfile
import re
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np


try:
    import tensorflow as tf  
except Exception:
    tf = None

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import sacrebleu
import spacy
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# ---------- Configuration ----------
EN_HI_MODEL_DIR = os.environ.get("EN_HI_MODEL", "backend/tf_model")  # local override
DEFAULT_EN_HI = "Helsinki-NLP/opus-mt-en-hi"
DEFAULT_HI_EN = "Helsinki-NLP/opus-mt-hi-en"

# ---------- Load models (tries local first, else HF hub) ----------
print("Loading models (this may take time on first run)...")

if os.path.isdir(EN_HI_MODEL_DIR) and os.listdir(EN_HI_MODEL_DIR):
    try:
        en_hi_tokenizer = AutoTokenizer.from_pretrained(EN_HI_MODEL_DIR)
        en_hi_model = TFAutoModelForSeq2SeqLM.from_pretrained(EN_HI_MODEL_DIR)
        print(f"Loaded local EN->HI model from {EN_HI_MODEL_DIR}")
    except Exception:
        print("Failed to load local model; falling back to HF hub.")
        en_hi_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EN_HI)
        en_hi_model = TFAutoModelForSeq2SeqLM.from_pretrained(DEFAULT_EN_HI)
else:
    en_hi_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EN_HI)
    en_hi_model = TFAutoModelForSeq2SeqLM.from_pretrained(DEFAULT_EN_HI)
    print(f"Loaded {DEFAULT_EN_HI} from Hugging Face hub")

# Load HI->EN model for back-translation
hi_en_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HI_EN)
hi_en_model = TFAutoModelForSeq2SeqLM.from_pretrained(DEFAULT_HI_EN)
print(f"Loaded {DEFAULT_HI_EN} for back-translation")

# spaCy NER
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("spaCy english model not present. Run: python -m spacy download en_core_web_sm to enable NER.")
    nlp = None

# ---------- Helper functions ----------
def translate_en_to_hi(text, max_length=128):
    inputs = en_hi_tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=max_length)
    outputs = en_hi_model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    decoded = en_hi_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded

def translate_hi_to_en(text, max_length=128):
    inputs = hi_en_tokenizer([text], return_tensors="tf", truncation=True, padding=True, max_length=max_length)
    outputs = hi_en_model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    decoded = hi_en_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return decoded

def extract_entities(text):
    if nlp is None:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

# Simple rule-based tone maps (small and illustrative)
FORMAL_MAP = {
    r"\bतू\b": "आप",
    r"\bतुम\b": "आप",
    r"\bकरो\b": "करिए",
}
INFORMAL_MAP = {
    r"\bआप\b": "तुम",
    r"\bकरिए\b": "करो",
}

def apply_tone(hindi_text, tone="normal"):
    if tone == "formal":
        for pat, sub in FORMAL_MAP.items():
            hindi_text = re.sub(pat, sub, hindi_text)
    elif tone == "informal":
        for pat, sub in INFORMAL_MAP.items():
            hindi_text = re.sub(pat, sub, hindi_text)
    return hindi_text

def bleu_score(reference, hypothesis):
    try:
        score = sacrebleu.sentence_bleu(hypothesis, [reference]).score
        return round(float(score), 2)
    except Exception:
        return None

# ---------- API endpoints ----------
@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Request JSON:
      { "text": "<english text>", "tone": "normal" | "formal" | "informal" }
    Response JSON:
      { "translation": "...", "back_translation": "...", "entities": [...], "bleu": 12.34 }
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    tone = data.get("tone", "normal")
    if not text:
        return jsonify({"error": "no text provided"}), 400

    # NER on original English (for frontend display)
    entities = extract_entities(text)

    # Translate EN -> HI
    hindi = translate_en_to_hi(text)
    hindi_toned = apply_tone(hindi, tone)

    # Back-translate HI -> EN
    back_en = translate_hi_to_en(hindi_toned)

    # BLEU (proxy, comparing back-translation to original English)
    bleu = bleu_score(text, back_en)

    return jsonify({
        "translation": hindi_toned,
        "back_translation": back_en,
        "entities": entities,
        "bleu": bleu
    })

@app.route("/api/backtranslate", methods=["POST"])
def api_backtranslate():
    """Translate Hindi -> English (separate endpoint if frontend wants it)."""
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text provided"}), 400
    back_en = translate_hi_to_en(text)
    return jsonify({"back_translation": back_en})

@app.route("/api/speak", methods=["POST"])
def api_speak():
    """
    Request JSON:
      { "text": "<hindi text to speak>" }
    Response:
      audio/mpeg stream (gTTS output)
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text provided"}), 400

    tmpf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmpf.name
    tmpf.close()

    try:
        tts = gTTS(text=text, lang="hi")
        tts.save(tmp_path)
        return send_file(tmp_path, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": f"tts failed: {str(e)}"}), 500

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True, "note": "backend alive"})

if __name__ == "__main__":
    # Run on 0.0.0.0 so it is reachable from other machines/containers if needed
    app.run(host="0.0.0.0", port=5000, debug=True)
