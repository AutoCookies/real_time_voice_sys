# routes/asr_ws.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from routes.messages_ws import broadcast
from routes.rooms import rooms
from pydub import AudioSegment
import whisper
import tempfile, os

router = APIRouter()

# Load Whisper model 1 lần khi server start
model = whisper.load_model("base")  # tiny/base/small/medium/large

@router.post("/audio_to_text/{room_id}/{client_id}")
async def audio_to_text(
    room_id: str,
    client_id: str,
    file: UploadFile = File(...),
    lang: str = Query(default="auto")
):
    suffix = os.path.splitext(file.filename)[-1] or ".webm"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp_file.name
    tmp_file.write(await file.read())
    tmp_file.close()

    wav_path = tmp_path + ".wav"
    try:
        # Convert WebM -> WAV mono
        audio = AudioSegment.from_file(tmp_path)
        audio = audio.set_channels(1)
        audio.export(wav_path, format="wav")

        # Whisper tự detect language nếu lang == "auto"
        language_option = None if lang == "auto" else lang
        result = model.transcribe(wav_path, language=language_option)
        text = result["text"].strip()
        print(f"[ASR] room={room_id}, client={client_id}, lang={lang}, text={text}")

        # Gửi text tới tất cả client trong room
        await broadcast(room_id, client_id, text, lang)


    except Exception as e:
        print("Transcribe error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return {"text": text}
