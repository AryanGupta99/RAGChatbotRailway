"""
FastAPI Chatbot Server for Zoho SalesIQ Integration via n8n
Provides webhook endpoint for interactive step-by-step support.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import urllib3
import uvicorn
from datetime import datetime

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Ace Cloud Hosting Support Bot", version="1.0.0")

# CORS middleware for n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "support-chatbot"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Get Pinecone index host
def get_index_host():
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json"
    }
    
    response = requests.get(
        "https://api.pinecone.io/indexes",
        headers=headers,
        verify=False
    )
    response.raise_for_status()
    
    indexes = response.json()
    for idx in indexes.get('indexes', []):
        if idx['name'] == INDEX_NAME:
            return idx['host']
    
    raise Exception(f"Index '{INDEX_NAME}' not found")

INDEX_HOST = get_index_host()
PINECONE_HEADERS = {
    "Api-Key": PINECONE_API_KEY,
    "Content-Type": "application/json"
}

# In-memory conversation storage (use Redis in production)
conversations: Dict[str, List[Dict]] = {}
# Store context for each session
session_contexts: Dict[str, str] = {}

# Request/Response Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str  # Unique ID from Zoho SalesIQ
    message: str
    conversation_history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    session_id: str
    response: str
    is_new_issue: bool
    retrieved_articles: Optional[List[str]] = []
    timestamp: str

def retrieve_context(query: str, top_k: int = 3):
    """
    Retrieve relevant documents from Pinecone.
    Prioritizes KB articles over chat transcripts.
    """
    # Generate query embedding
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # First, try to get KB articles
    url = f"https://{INDEX_HOST}/query"
    kb_payload = {
        "vector": query_embedding,
        "topK": 5,
        "includeMetadata": True,
        "filter": {"source": {"$eq": "kb_article"}}
    }
    
    kb_response = requests.post(
        url,
        headers=PINECONE_HEADERS,
        json=kb_payload,
        verify=False
    )
    kb_response.raise_for_status()
    kb_results = kb_response.json().get('matches', [])
    
    # Filter KB articles with good scores (>0.4)
    good_kb_articles = [m for m in kb_results if m['score'] > 0.4]
    
    # If we have good KB articles, use them
    if good_kb_articles:
        print(f"[Context] Using {len(good_kb_articles)} KB articles")
        return good_kb_articles[:top_k]
    
    # Otherwise, fall back to all sources
    print(f"[Context] No good KB articles found, using all sources")
    all_payload = {
        "vector": query_embedding,
        "topK": top_k,
        "includeMetadata": True
    }
    
    all_response = requests.post(
        url,
        headers=PINECONE_HEADERS,
        json=all_payload,
        verify=False
    )
    all_response.raise_for_status()
    
    return all_response.json().get('matches', [])

def is_new_issue(message: str, history: List[Dict]) -> bool:
    """Determine if this is a new issue or continuation."""
    if len(history) == 0:
        return True
    
    message_lower = message.lower()
    
    # Simple continuation keywords (1-3 words)
    continuation_keywords = ["yes", "done", "completed", "next", "ok", "okay", "continue"]
    
    # If message is ONLY a continuation keyword, it's a continuation
    if message_lower.strip() in continuation_keywords:
        return False
    
    # If message is very short (1-3 words) and contains continuation keywords
    if len(message.split()) <= 3:
        if any(keyword in message_lower for keyword in continuation_keywords):
            return False
    
    # If message mentions a specific technical issue/topic, it's a NEW issue
    technical_keywords = [
        "quickbooks", "frozen", "error", "issue", "problem", "not working",
        "disk space", "password", "reset", "server", "lacerte", "drake",
        "outlook", "email", "printer", "install", "setup", "configure"
    ]
    
    if any(keyword in message_lower for keyword in technical_keywords):
        return True
    
    # Long messages (>5 words) are usually new issues
    if len(message.split()) > 5:
        return True
    
    return False

def build_context(context_docs: List[Dict]) -> str:
    """Build context string from retrieved documents, filtering out bad chat transcripts."""
    context_parts = []
    
    for doc in context_docs:
        source = doc['metadata']['source']
        text = doc['metadata'].get('text', '')
        
        # Skip chat transcripts with placeholders
        if source == 'chat_transcript' and ('[EMAIL]' in text or '[URL]' in text or '[PHONE]' in text):
            continue
        
        if source == 'kb_article':
            title = doc['metadata'].get('title', 'KB Article')
            context_parts.append(f"[KB Article] {title}\n{text}")
        else:
            context_parts.append(f"[Support Chat]\n{text}")
    
    return "\n\n---\n\n".join(context_parts)

def generate_response(message: str, history: List[Dict], context: Optional[str] = None) -> str:
    """Generate response using GPT-4o-mini."""
    
    system_prompt = """You are AceBuddy, a technical support assistant for Ace Cloud Hosting.

YOUR JOB: Guide users through the EXACT steps from the knowledge base, in order, 1-2 steps at a time.

CRITICAL RULES:
1. ALWAYS give steps in sequential order: 1-2, then 3-4, then 5, etc.
2. NEVER skip steps - if user says "done" after steps 1-2, give steps 3-4 next
3. COPY the exact step text from the KB - do not paraphrase
4. After giving 1-2 steps, ALWAYS ask "Have you completed this?"
5. DO NOT make up steps that aren't in the knowledge base
6. Look at your previous response to see which steps you already gave, then give the NEXT steps
7. If no relevant KB article is found, offer to connect them with support at 1-888-415-5240

EXAMPLE CONVERSATION:
KB: "Step 1: Visit portal. Step 2: Click Forgot. Step 3: Enter CAPTCHA. Step 4: Choose method. Step 5: Enter password."

Bot: "Step 1: Visit portal. Step 2: Click Forgot. Have you completed this?"
User: "done"
Bot: "Step 3: Enter CAPTCHA. Step 4: Choose method. Have you completed this?"
User: "yes"  
Bot: "Step 5: Enter password. Have you completed this?"

NEVER skip from step 2 to step 5!"""
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add context if available (for both new issues AND continuations)
    if context and context.strip():
        context_message = f"""KNOWLEDGE BASE ARTICLE (follow these steps IN ORDER):

{context}

IMPORTANT: 
- Give steps 1-2 first
- When user says "done", give steps 3-4 next (NOT step 5!)
- Continue sequentially until all steps are complete
- Copy the exact step text, don't paraphrase"""
        messages.append({"role": "system", "content": context_message})
    elif not context or not context.strip():
        # No good context found
        return "I don't have specific steps for this issue in my knowledge base. Please contact our support team at 1-888-415-5240 or support@acecloudhosting.com for assistance."
    
    # Add conversation history
    messages.extend(history)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Generate response
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )
    
    return response.choices[0].message.content

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Ace Cloud Hosting Support Bot",
        "version": "1.0.0",
        "endpoints": {
            "salesiq_webhook": "/webhook/salesiq (Use this for SalesIQ)",
            "chat": "/chat (For n8n integration)",
            "reset": "/reset/{session_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {
        "status": "healthy",
        "pinecone": "connected",
        "openai": "connected",
        "active_sessions": len(conversations)
    }

@app.post("/webhook/salesiq")
async def salesiq_webhook(request: dict):
    """
    Direct webhook endpoint for Zoho SalesIQ.
    Handles SalesIQ's webhook format and returns response in the exact format SalesIQ expects.
    
    Expected Input:
    {
        "session_id": "abc123",
        "message": {"text": "user message"}
    }
    
    Required Output:
    {
        "action": "reply",
        "replies": ["bot response"],
        "session_id": "abc123"
    }
    """
    try:
        # Log incoming request for debugging
        print(f"[SalesIQ] Received: {request}")
        
        # Extract session_id from SalesIQ's nested structure
        visitor = request.get('visitor', {})
        session_id = (
            visitor.get('active_conversation_id') or 
            request.get('session_id') or 
            request.get('visitor', {}).get('id') or
            'unknown'
        )
        
        # Extract message text from nested structure
        message_obj = request.get('message', {})
        message_text = message_obj.get('text', '') if isinstance(message_obj, dict) else str(message_obj)
        
        # Handle empty messages (greeting)
        if not message_text or message_text.strip() == '':
            print(f"[SalesIQ] Empty message, sending greeting")
            return {
                "action": "reply",
                "replies": ["Hi! I'm AceBuddy, your Ace Cloud Hosting support assistant. What can I help you with today?"],
                "session_id": session_id
            }
        
        print(f"[SalesIQ] Session: {session_id}, Message: {message_text}")
        
        # Get or create conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        history = conversations[session_id]
        
        # Handle simple greetings without RAG
        message_lower = message_text.lower().strip()
        greeting_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'hi there', 'hello there', 'hey there', 'greetings', 'howdy'
        ]
        
        # Check if message is ONLY a greeting (no technical question)
        is_greeting = (
            message_lower in greeting_patterns or
            (len(message_text.split()) <= 3 and any(g in message_lower for g in ['hello', 'hi', 'hey', 'morning', 'afternoon', 'evening']))
        )
        
        if is_greeting and len(history) == 0:
            print(f"[SalesIQ] Simple greeting detected")
            return {
                "action": "reply",
                "replies": ["Hello! How can I assist you today?"],
                "session_id": session_id
            }
        
        # Handle contact/support requests directly
        contact_keywords = ['email', 'phone', 'contact', 'support number', 'call', 'reach', 'telephone']
        if any(keyword in message_lower for keyword in contact_keywords):
            # Check if they're asking for contact info (not a technical issue)
            if not any(tech in message_lower for tech in ['error', 'issue', 'problem', 'not working', 'frozen', 'crash']):
                print(f"[SalesIQ] Contact request detected")
                return {
                    "action": "reply",
                    "replies": ["You can reach Ace Cloud Hosting support at:\n\nPhone: 1-888-415-5240 (24/7)\nEmail: support@acecloudhosting.com\n\nHow else can I help you?"],
                    "session_id": session_id
                }
        
        # Determine if new issue
        new_issue = is_new_issue(message_text, history)
        
        # If it's a new issue, clear old context
        if new_issue and session_id in session_contexts:
            print(f"[SalesIQ] New issue detected, clearing old context")
            del session_contexts[session_id]
        
        context = None
        
        # If new issue, retrieve context from Pinecone
        if new_issue:
            print(f"[SalesIQ] New issue detected, retrieving context...")
            try:
                context_docs = retrieve_context(message_text, top_k=3)
                if context_docs:
                    context = build_context(context_docs)
                    # Store context for this session
                    session_contexts[session_id] = context
                    print(f"[SalesIQ] Retrieved {len(context_docs)} context documents")
            except Exception as e:
                print(f"[SalesIQ] Context retrieval failed: {str(e)}")
                # Continue without context
        else:
            # Continuation - use stored context if available
            context = session_contexts.get(session_id)
            if context:
                print(f"[SalesIQ] Using stored context for continuation")
        
        # Generate response
        print(f"[SalesIQ] Calling OpenAI...")
        if context:
            print(f"[SalesIQ] Context preview: {context[:300]}...")
        response_text = generate_response(message_text, history, context)
        
        # Clean response - remove markdown and extra line breaks
        response_text = response_text.replace('**', '').replace('*', '').strip()
        
        # Remove excessive line breaks (replace double/triple newlines with single)
        import re
        response_text = re.sub(r'\n\s*\n+', '\n', response_text)
        
        # Don't truncate - let full response through
        # SalesIQ can handle longer messages
        
        print(f"[SalesIQ] Response: {response_text}")
        
        # Update conversation history
        conversations[session_id].append({"role": "user", "content": message_text})
        conversations[session_id].append({"role": "assistant", "content": response_text})
        
        # Return in EXACT SalesIQ expected format
        return {
            "action": "reply",
            "replies": [response_text],
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"[SalesIQ] ERROR: {str(e)}")
        # Return error in SalesIQ format
        return {
            "action": "reply",
            "replies": ["I'm having technical difficulties. Please call our support team at 1-888-415-5240."],
            "session_id": request.get('session_id', 'unknown')
        }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for n8n webhook.
    
    Receives message from Zoho SalesIQ via n8n, processes it, and returns response.
    """
    try:
        session_id = request.session_id
        message = request.message
        
        # Get or create conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        history = conversations[session_id]
        
        # Determine if new issue
        new_issue = is_new_issue(message, history)
        
        context = None
        retrieved_articles = []
        
        # If new issue, retrieve context from Pinecone
        if new_issue:
            context_docs = retrieve_context(message, top_k=3)
            
            if context_docs:
                context = build_context(context_docs)
                retrieved_articles = [
                    doc['metadata'].get('title', 'KB Article')
                    for doc in context_docs
                    if doc['metadata']['source'] == 'kb_article'
                ]
        
        # Generate response
        response_text = generate_response(message, history, context)
        
        # Update conversation history
        conversations[session_id].append({"role": "user", "content": message})
        conversations[session_id].append({"role": "assistant", "content": response_text})
        
        # Return response
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            is_new_issue=new_issue,
            retrieved_articles=retrieved_articles if retrieved_articles else None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{session_id}")
async def reset_conversation(session_id: str):
    """Reset conversation for a session."""
    reset_count = 0
    if session_id in conversations:
        del conversations[session_id]
        reset_count += 1
    if session_id in session_contexts:
        del session_contexts[session_id]
        reset_count += 1
    
    if reset_count > 0:
        return {"status": "success", "message": f"Conversation {session_id} reset"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions (for debugging)."""
    return {
        "active_sessions": len(conversations),
        "sessions": list(conversations.keys())
    }

if __name__ == "__main__":
    # Get port from environment variable (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    
    print("="*70)
    print("ACE CLOUD HOSTING - SUPPORT BOT SERVER")
    print("="*70)
    print(f"\n🚀 Starting FastAPI server on port {port}...")
    print(f"📍 Endpoint: http://0.0.0.0:{port}")
    print(f"📖 Docs: http://0.0.0.0:{port}/docs")
    print("\n✅ Ready to receive webhooks from n8n!")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
