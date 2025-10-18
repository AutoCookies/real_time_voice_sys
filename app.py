from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import messages_ws, asr_ws
app = FastAPI(title="Whisper Voice-to-Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(messages_ws.router)
app.include_router(asr_ws.router)

@app.get("/")
def root():
    return {"message": "Server is running!"}
