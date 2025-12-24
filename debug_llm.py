from core.bot import generate_llm_response
import logging

# Setup basic logging to see errors
logging.basicConfig(level=logging.ERROR)

print("Attempting to call LLM...")
response = generate_llm_response("Hello", [])
print(f"Response: {response}")
