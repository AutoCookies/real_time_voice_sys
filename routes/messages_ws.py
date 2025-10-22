from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from routes.rooms import rooms
import torch
import re
import asyncio

router = APIRouter(prefix="/ws")

# =====================================================
#                 LOAD TRANSLATION MODELS
# =====================================================
print("🚀 Loading translation models...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_load(func, fallback=None):
    try:
        return func()
    except Exception as e:
        print(f"⚠️ Model load failed: {e}")
        return fallback

# --- JA <-> EN ---
ja_to_en = safe_load(lambda: pipeline("translation", model="Mitsua/elan-mt-bt-ja-en", device=0 if torch.cuda.is_available() else -1))
en_to_ja = safe_load(lambda: pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap", device=0 if torch.cuda.is_available() else -1))

# --- VI <-> EN ---
vietai_model_name = "VietAI/envit5-translation"
vietai_tokenizer, vietai_model = safe_load(lambda: (
    AutoTokenizer.from_pretrained(vietai_model_name),
    AutoModelForSeq2SeqLM.from_pretrained(vietai_model_name).to(device)
), (None, None))

print("✅ All translation models loaded.\n")

# =====================================================
#                TRANSLATION HELPERS
# =====================================================
def clean_output(text: str) -> str:
    """Clean repetitive or overly long text"""
    text = text.strip()
    if re.fullmatch(r"(\b\w+\b[, ]*){5,}", text):
        words = re.findall(r"\b\w+\b", text)
        if len(set(words)) == 1:
            text = words[0]
    return text[:300].strip()

def seq2seq_translate(prefix: str, text: str) -> str:
    """Translate using VietAI envit5"""
    if not vietai_model:
        return text
    inputs = vietai_tokenizer(f"{prefix}: {text}", return_tensors="pt", truncation=True, padding=True).to(vietai_model.device)
    outputs = vietai_model.generate(**inputs, max_new_tokens=256)
    return clean_output(vietai_tokenizer.decode(outputs[0], skip_special_tokens=True))

def pipeline_translate(model, text: str) -> str:
    if not model:
        return text
    try:
        return clean_output(model(text, max_length=256)[0]["translation_text"])
    except Exception:
        return text

async def safe_async_translate(func, *args):
    """Run translation safely in thread with timeout"""
    try:
        return await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=10)
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return args[-1]

# --- Unified translator ---
async def translate(text: str, src: str, tgt: str):
    if src == tgt:
        return text

    # pivot → English
    if tgt == "en":
        if src == "vi": return await safe_async_translate(seq2seq_translate, "vi", text)
        if src == "ja": return await safe_async_translate(pipeline_translate, ja_to_en, text)
        return text

    # pivot ← English
    if src == "en":
        if tgt == "vi": return await safe_async_translate(seq2seq_translate, "en", text)
        if tgt == "ja": return await safe_async_translate(pipeline_translate, en_to_ja, text)
        return text

    # non-English pair via English pivot
    to_en = await translate(text, src, "en")
    return await translate(to_en, "en", tgt)

# =====================================================
#                WEBSOCKET HANDLER
# =====================================================
@router.websocket("/{room_id}/{client_id}/{lang}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str, lang: str):
    await websocket.accept()
    rooms.setdefault(room_id, []).append((websocket, lang))
    print(f"🟢 [JOIN] {client_id} ({lang}) → room={room_id}")

    try:
        while True:
            text = await websocket.receive_text()
            await broadcast(room_id, client_id, text, lang)
    except WebSocketDisconnect:
        print(f"🔴 [LEAVE] {client_id} ({lang})")
    except Exception as e:
        print(f"⚠️ WebSocket error from {client_id}: {e}")
    finally:
        rooms[room_id] = [x for x in rooms[room_id] if x[0].client_state.value == 1]

# =====================================================
#                BROADCAST WITH PIVOT EN
# =====================================================
async def broadcast(room_id: str, client_id: str, text: str, src_lang: str):
    if room_id not in rooms: return
    print(f"\n[{client_id}] ({src_lang}): {text}")

    # pivot to English
    pivot_en = await translate(text, src_lang, "en")
    print(f"🔹 Pivot (EN): {pivot_en}")

    for ws, tgt_lang in list(rooms[room_id]):
        if ws.client_state.value != 1:
            continue
        try:
            msg = await translate(pivot_en, "en", tgt_lang)
            await ws.send_text(f"{client_id} ({src_lang}): {msg}")
            print(f"📤 [{src_lang} → {tgt_lang}] {msg}")
        except Exception as e:
            print(f"⚠️ Broadcast error ({tgt_lang}): {e}")

    print("✅ Broadcast complete\n")
