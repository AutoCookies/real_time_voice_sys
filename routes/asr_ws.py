from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from routes.messages_ws import rooms  # dict chứa WebSocket rooms
from faster_whisper import WhisperModel
from routes.messages_ws import broadcast
import tempfile, os

router = APIRouter()

# Load model small 1 lần khi server start
model = WhisperModel("small", device="cpu")  # hoặc "cuda" nếu có GPU

@router.post("/audio_to_text/{room_id}/{client_id}")
async def audio_to_text(
    room_id: str,
    client_id: str,
    file: UploadFile = File(...),
    lang: str = Query(default="auto")  # nhận query param lang
):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp_file.name
    tmp_file.write(await file.read())
    tmp_file.close()

    try:
        language_option = None if lang == "auto" else lang
        segments, info = model.transcribe(tmp_path, beam_size=5, language=language_option)
        text = " ".join([seg.text for seg in segments])
        print(f"[ASR] room={room_id}, client={client_id}, lang={lang}, text={text}")

        # Chỉ gọi broadcast → tự dịch & gửi cho từng client
        await broadcast(room_id, client_id, text, lang)

    except Exception as e:
        print("Transcribe error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return {"text": text}