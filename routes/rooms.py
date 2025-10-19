# routes/rooms.py
from fastapi import WebSocket
from typing import Dict, List, Tuple

rooms: Dict[str, List[WebSocket]] = {}