from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routes.rooms import rooms

router = APIRouter()

async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str, lang: str = "vi"):
    await websocket.accept()
    if room_id not in rooms:
        rooms[room_id] = []
    rooms[room_id].append((websocket, lang))
    try:
        while True:
            data = await websocket.receive_text()
            # Gọi broadcast để dịch & gửi tới các client khác
            await broadcast(room_id, client_id, data, lang)
    except WebSocketDisconnect:
        rooms[room_id] = [x for x in rooms[room_id] if x[0] != websocket]

# --- Broadcast helper (dịch) ---
from transformers import pipeline

translator_vi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")
translator_en_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ja")
translator_ja_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
translator_en_vi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi")

async def broadcast(room_id: str, client_id: str, text_original: str, original_lang: str):
    if room_id not in rooms:
        return
    for ws, client_lang in rooms[room_id]:
        try:
            if client_lang == original_lang:
                text_to_send = text_original
            else:
                # Dịch text qua English pivot
                if original_lang == "vi" and client_lang == "ja":
                    en = translator_vi_en(text_original)[0]['translation_text']
                    text_to_send = translator_en_ja(en)[0]['translation_text']
                elif original_lang == "ja" and client_lang == "vi":
                    en = translator_ja_en(text_original)[0]['translation_text']
                    text_to_send = translator_en_vi(en)[0]['translation_text']
                else:
                    text_to_send = text_original
            await ws.send_text(f"{client_id}: {text_to_send}")
        except:
            pass
