import os
import logging
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseModel):
    # App Settings
    APP_TITLE: str = "Ace Cloud Hosting Support Bot - Hybrid"
    APP_VERSION: str = "2.0.0"
    PORT: int = int(os.getenv("PORT", 8000))
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = "gpt-4o-mini"
    
    # Zoho Settings
    SALESIQ_WIDGET_CODE: str = os.getenv("SALESIQ_WIDGET_CODE", "")
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROMPTS_DIR: str = os.path.join(BASE_DIR, "prompts")

@lru_cache()
def get_settings():
    return Settings()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Reduce noise from libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)

settings = get_settings()
