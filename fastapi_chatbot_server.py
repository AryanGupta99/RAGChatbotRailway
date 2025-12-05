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
    
    continuation_keywords = [
        "yes", "done", "completed", "next", "continue", "ok", "okay",
        "no", "didn't work", "error", "failed", "stuck", "help"
    ]
    
    message_lower = message.lower()
    
    # Short responses are usually continuations
    if len(message.split()) <= 5:
        for keyword in continuation_keywords:
            if keyword in message_lower:
                return False
    
    # Long questions about different topics are new issues
    if any(word in message_lower for word in ["how", "what", "why", "can", "fix", "error", "issue"]):
        if len(message.split()) > 5:
            return True
    
    return False

def build_context(context_docs: List[Dict]) -> str:
    """Build context string from retrieved documents."""
    context_parts = []
    
    for doc in context_docs:
        source = doc['metadata']['source']
        
        if source == 'kb_article':
            title = doc['metadata'].get('title', 'KB Article')
            context_parts.append(f"[KB Article] {title}\n{doc['metadata']['text']}")
        else:
            context_parts.append(f"[Support Chat]\n{doc['metadata']['text']}")
    
    return "\n\n---\n\n".join(context_parts)

def generate_response(message: str, history: List[Dict], context: Optional[str] = None) -> str:
    """Generate response using GPT-4o-mini."""
    
    system_prompt = """You are AceBuddy, a technical support assistant for Ace Cloud Hosting.

YOUR ONLY JOB: Read the numbered steps from the knowledge base and guide the user through them ONE OR TWO STEPS AT A TIME.

RULES:
1. Give ONLY steps 1-2 first, then ask "Have you completed this?"
2. When user says "done" or "yes", give steps 3-4, then ask again
3. Continue until all steps are complete
4. COPY the step text EXACTLY - do not rewrite or add anything
5. DO NOT make up steps that aren't in the knowledge base
6. Keep responses under 200 characters

EXAMPLE:
KB has: "Step 1: Visit portal. Step 2: Click Forgot Password. Step 3: Enter username."
You say: "Step 1: Visit portal. Step 2: Click Forgot Password. Done?"
User: "done"
You say: "Step 3: Enter username. Done?"

DO NOT add steps like "check email" or "follow instructions" unless they're in the KB."""
    
    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add context if available (for both new issues AND continuations)
    if context:
        context_message = f"""KNOWLEDGE BASE STEPS (copy these EXACTLY):

{context}

Remember: Give steps 1-2 first. When user confirms, give steps 3-4. Copy the exact wording."""
        messages.append({"role": "system", "content": context_message})
    
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
        
        # Determine if new issue
        new_issue = is_new_issue(message_text, history)
        
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
        
        # Clean response - remove markdown, keep it short
        response_text = response_text.replace('**', '').replace('*', '').strip()
        
        # Limit to 200 characters for SalesIQ
        if len(response_text) > 200:
            response_text = response_text[:197] + "..."
        
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
