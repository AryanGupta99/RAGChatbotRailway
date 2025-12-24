import logging
import traceback
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
from openai import OpenAI

from core.config import settings
from core.prompts import load_expert_prompt
from integrations.zoho import ZohoSalesIQAPI, ZohoDeskAPI

logger = logging.getLogger(__name__)

# Initialize services
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
salesiq_api = ZohoSalesIQAPI()
desk_api = ZohoDeskAPI()

# In-Memory State (as requested)
conversations: Dict[str, List[Dict]] = {}

def get_conversation_history(session_id: str) -> List[Dict]:
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]

def clear_conversation_history(session_id: str):
    if session_id in conversations:
        del conversations[session_id]

def generate_llm_response(message: str, history: List[Dict]) -> str:
    """Generate response using LLM with embedded resolution steps"""
    system_prompt = load_expert_prompt()
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    try:
        response = openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return "I'm having trouble processing that request right now. Please try again."

def is_acknowledgment_message(msg: str) -> bool:
    msg = msg.lower().strip()
    if 'then' in msg: return False
    direct_acks = ["okay", "ok", "thanks", "thank you", "got it", "understood", "alright"]
    if msg in direct_acks: return True
    thanks_patterns = ["thank", "thnk", "thx", "ty"]
    if any(pattern in msg for pattern in thanks_patterns) and len(msg) < 20:
        return True
    return False

async def handle_salesiq_message(request: Dict) -> Dict:
    """Main handler for SalesIQ webhooks"""
    session_id = 'unknown'
    try:
        # Extract Session ID
        visitor = request.get('visitor', {})
        chat = request.get('chat', {})
        conversation = request.get('conversation', {})
        
        session_id = (
            visitor.get('active_conversation_id') or
            chat.get('id') or
            conversation.get('id') or
            request.get('session_id') or 
            visitor.get('id') or
            'unknown'
        )
        
        # Extract Message
        message_obj = request.get('message', {})
        message_text = (message_obj.get('text', '') if isinstance(message_obj, dict) else str(message_obj)).strip()
        payload = request.get('payload', '')
        
        if not message_text:
            return {"action": "reply", "replies": ["Hi! I'm AceBuddy. How can I help?"], "session_id": session_id}
            
        history = get_conversation_history(session_id)
        message_lower = message_text.lower().strip()

        # --- 1. Intent Detection: Greetings ---
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning']
        is_greeting = (message_lower in greeting_patterns or (len(message_text.split()) <= 3 and any(g in message_lower for g in greeting_patterns)))
        
        if is_greeting and len(history) == 0:
            return {"action": "reply", "replies": ["Hello! How can I assist you today?"], "session_id": session_id}

        # --- 2. Intent Detection: Contact Info ---
        if any(p in message_lower for p in ['support email', 'support number', 'contact support']):
            return {
                "action": "reply", 
                "replies": ["Phone: 1-888-415-5240 (24/7)\nEmail: support@acecloudhosting.com"], 
                "session_id": session_id
            }
            
        # --- 3. Intent Detection: Human Agent Handoff ---
        # Logic: If user says "yes/ok" to a bot message proposing a transfer
        if len(history) > 0 and ('yes' in message_lower or 'ok' in message_lower or 'connect' in message_lower):
            last_bot_message = history[-1].get('content', '') if history[-1].get('role') == 'assistant' else ''
            if 'human agent' in last_bot_message.lower():
                # Perform Transfer
                conversation_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history])
                salesiq_api.create_chat_session(session_id, conversation_text)
                clear_conversation_history(session_id)
                return {
                    "action": "reply", 
                    "replies": ["Connecting you to support... (Call 1-888-415-5240 if needed)"], 
                    "session_id": session_id
                }

        # --- 4. Intent Detection: Resolved ---
        if any(k in message_lower for k in ["resolved", "fixed", "working now"]):
            msg = "Great! Glad to hear it."
            history.append({"role": "user", "content": message_text})
            history.append({"role": "assistant", "content": msg})
            salesiq_api.close_chat(session_id, "resolved")
            clear_conversation_history(session_id)
            return {"action": "reply", "replies": [msg], "session_id": session_id}

        # --- 5. Intent Detection: Not Resolved (Options) ---
        if any(k in message_lower for k in ["not resolved", "not fixed", "still not"]):
            msg = "I understand. Here are 3 options:"
            history.append({"role": "user", "content": message_text})
            history.append({"role": "assistant", "content": msg})
            return {
                "action": "reply",
                "replies": [msg],
                "suggestions": [
                    {"text": "📞 Instant Chat", "action_type": "reply", "action_value": "1"},
                    {"text": "📅 Schedule Callback", "action_type": "reply", "action_value": "2"},
                    {"text": "🎫 Create Ticket", "action_type": "reply", "action_value": "3"}
                ],
                "session_id": session_id
            }

        # --- 6. Intent Detection: Option Selection ---
        if "instant chat" in message_lower or "1" == message_lower or payload == "option_1":
            # Transfer Logic
            conversation_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history])
            
            # Using visitor email is safer if available
            visitor_email = visitor.get('email', 'support@acecloudhosting.com') if isinstance(visitor, dict) else 'support@acecloudhosting.com'
            
            salesiq_api.create_chat_session(visitor_email, conversation_text, visitor_info=visitor)
            clear_conversation_history(session_id)
            return {"action": "reply", "replies": ["Connecting you to support..."], "session_id": session_id}
            
        if "callback" in message_lower or "2" == message_lower or payload == "option_2":
            reply = "I've created a callback request. Our team will call you shortly."
            desk_api.create_callback_ticket("support@acecloudhosting.com", "pending", "ASAP", description="From Chat")
            salesiq_api.close_chat(session_id, "callback_scheduled")
            clear_conversation_history(session_id)
            return {"action": "reply", "replies": [reply], "session_id": session_id}

        if "ticket" in message_lower or "3" == message_lower or payload == "option_3":
            reply = "I've created a support ticket. Reference: TK-NEW."
            desk_api.create_support_ticket("Support Request", "From Chat", "support@acecloudhosting.com")
            salesiq_api.close_chat(session_id, "ticket_created")
            clear_conversation_history(session_id)
            return {"action": "reply", "replies": [reply], "session_id": session_id}

        # --- 7. Intent Detection: Password Reset (Smart Routing) ---
        if any(k in message_lower for k in ["password", "reset", "forgot"]):
            # Check context
            last_bot_msg = history[-1].get('content', '').lower() if history else ''
            if 'selfcare' in last_bot_msg:
                if 'yes' in message_lower or 'registered' in message_lower:
                    response = "Great! Visit https://selfcare.acecloudhosting.com/forgot"
                else:
                    response = "Please contact support at 1-888-415-5240 for manual reset."
            else:
                response = "I can help! Are you registered on the SelfCare portal?"
            
            history.append({"role": "user", "content": message_text})
            history.append({"role": "assistant", "content": response})
            return {"action": "reply", "replies": [response], "session_id": session_id}

        # --- 8. Standard LLM Flow ---
        # Acknowledgment Check
        is_ack = is_acknowledgment_message(message_text)
        is_troubleshooting = False
        if history:
            last_msg = history[-1].get('content', '').lower()
            if any(k in last_msg for k in ['step', 'click', 'press', 'type']):
                is_troubleshooting = True
        
        if is_ack and not is_troubleshooting:
             return {"action": "reply", "replies": ["Is there anything else?"], "session_id": session_id}
             
        # Generate Response
        response_text = generate_llm_response(message_text, history)
        
        # Cleanup response (formatting)
        response_text = response_text.replace('**', '').strip()
        
        history.append({"role": "user", "content": message_text})
        history.append({"role": "assistant", "content": response_text})
        
        return {"action": "reply", "replies": [response_text], "session_id": session_id}

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        logger.error(traceback.format_exc())
        return {"action": "reply", "replies": ["System error. Please call 1-888-415-5240."], "session_id": session_id}
