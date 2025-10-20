from fastapi import WebSocket
from typing import Dict, List, Tuple

# rooms: key=room_id, value = list of (WebSocket, lang)
# lang: 'vi' | 'ja' | 'en' ...
rooms: Dict[str, List[Tuple[WebSocket, str]]] = {}