from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from core.config import settings
from core.bot import handle_salesiq_message, conversations, clear_conversation_history

# Setup logging
from core.config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "endpoints": {
            "salesiq_webhook": "/webhook/salesiq",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "mode": "production",
        "active_sessions": len(conversations),
    }

@app.post("/webhook/salesiq")
async def salesiq_webhook_handler(request: Request):
    """Entry point for Zoho SalesIQ Webhook"""
    try:
        data = await request.json()
        logger.info(f"Webhook received: {data.keys()}")
        return await handle_salesiq_message(data)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"action": "reply", "replies": ["Internal Error. Please contact support."]}

@app.get("/webhook/salesiq")
async def salesiq_webhook_verify():
    """Verification for Zoho SalesIQ"""
    return {"status": "ready", "message": "Webhook functional"}


@app.post("/reset/{session_id}")
async def reset_session_handler(session_id: str):
    clear_conversation_history(session_id)
    return {"status": "success", "message": f"Session {session_id} reset"}

if __name__ == "__main__":
    print(f"Starting server on port {settings.PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
