import os
import logging
from core.config import settings

logger = logging.getLogger(__name__)

def load_expert_prompt() -> str:
    """Load the expert prompt from the markdown file."""
    prompt_path = os.path.join(settings.PROMPTS_DIR, "expert_prompt.md")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load expert prompt from {prompt_path}: {e}")
        # Fallback to a minimal prompt if file is missing (failsafe)
        return "You are a helpful support assistant."
