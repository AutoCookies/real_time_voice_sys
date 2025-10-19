from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from routes.rooms import rooms
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

router = APIRouter(prefix="/ws")

ja_to_en = pipeline('translation', model='Mitsua/elan-mt-bt-ja-en')
en_to_ja = pipeline('translation', model="Helsinki-NLP/opus-mt-en-jap")

model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
vie_to_en = AutoModelForSeq2SeqLM.from_pretrained(model_name) # also english to vietnamese


@router.websocket("/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    if room_id not in rooms:
        rooms[room_id] = []
    rooms[room_id].append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            for ws in rooms[room_id]:
                await ws.send_text(f"{client_id}: {data}")
    except WebSocketDisconnect:
        rooms[room_id] = [ws for ws in rooms[room_id] if ws != websocket]

# Broadcast helper
async def broadcast(room_id: str, client_id: str, text: str):
    if room_id not in rooms:
        return
    for ws in rooms[room_id]:
        try:
            await ws.send_text(f"{client_id}: {text}")
        except:
            pass
