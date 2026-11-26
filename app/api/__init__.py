"""HTTP 路由层。"""

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.debug import router as debug_router
from app.api.health import router as health_router

__all__ = ["auth_router", "chat_router", "debug_router", "health_router"]
