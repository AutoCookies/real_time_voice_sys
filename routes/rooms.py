from fastapi import WebSocket
from typing import Dict, List, Tuple

# rooms dict chung giữa websocket và API
rooms: Dict[str, List[Tuple[WebSocket, str]]] = {}
