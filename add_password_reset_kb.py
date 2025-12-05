"""
Add the password reset KB article to Pinecone
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import urllib3
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "support-chatbot"
EMBEDDING_MODEL = "text-embedding-3-small"

# Get index host
headers = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
response = requests.get("https://api.pinecone.io/indexes", headers=headers, verify=False)
indexes = response.json()
INDEX_HOST = None
for idx in indexes.get('indexes', []):
    if idx['name'] == INDEX_NAME:
        INDEX_HOST = idx['host']
        break

# Create the KB article
kb_article = {
    "title": "How to reset server password using Self-Care Portal",
    "text": """How to reset server password using Self-Care Portal

Issue: Password reset process using self care

First you need to be registered on self care portal.

To reset the password using Selfcare Portal, please follow the simple steps outlined below:

Step 1: Visit Selfcare Portal https://selfcare.acecloudhosting.com Click "Forgot your password".

Step 2: Enter your Server Username.

Step 3: Enter the CAPTCHA verification and Click Continue.

Step 4: In the window that opens, choose an authentication method from the list.

Step 5: Enter your new password and click Reset to finish.

Benefits:
- Reset your password anytime without contacting support
- Secure authentication methods
- Quick and easy process
- Available 24/7"""
}

print("Creating KB article for password reset...")
print(f"Title: {kb_article['title']}")
print(f"Text length: {len(kb_article['text'])} characters")

# Generate embedding
print("\nGenerating embedding...")
response = openai_client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=kb_article['text']
)
embedding = response.data[0].embedding

# Prepare vector for Pinecone
vector = {
    "id": "kb_password_reset_selfcare",
    "values": embedding,
    "metadata": {
        "source": "kb_article",
        "title": kb_article['title'],
        "text": kb_article['text']
    }
}

# Upsert to Pinecone
print("\nUpserting to Pinecone...")
url = f"https://{INDEX_HOST}/vectors/upsert"
payload = {
    "vectors": [vector]
}

response = requests.post(
    url,
    headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
    json=payload,
    verify=False
)

if response.status_code == 200:
    print("✅ SUCCESS! KB article added to Pinecone")
    print(f"Vector ID: {vector['id']}")
else:
    print(f"❌ FAILED: {response.status_code}")
    print(response.text)

# Test retrieval
print("\n" + "="*70)
print("TESTING RETRIEVAL")
print("="*70)

test_queries = [
    "help me reset my password",
    "password reset selfcare",
    "how to reset server password",
    "forgot my password"
]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    
    # Generate query embedding
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Search with KB filter
    search_url = f"https://{INDEX_HOST}/query"
    search_payload = {
        "vector": query_embedding,
        "topK": 3,
        "includeMetadata": True,
        "filter": {"source": {"$eq": "kb_article"}}
    }
    
    response = requests.post(
        search_url,
        headers={"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"},
        json=search_payload,
        verify=False
    )
    
    results = response.json().get('matches', [])
    if results:
        top_result = results[0]
        print(f"  Top result: {top_result['metadata'].get('title', 'N/A')}")
        print(f"  Score: {top_result['score']:.4f}")
        if top_result['id'] == 'kb_password_reset_selfcare':
            print("  ✅ Correct article retrieved!")
        else:
            print("  ⚠️ Different article retrieved")
    else:
        print("  ❌ No results found")

print("\n" + "="*70)
print("DONE! The password reset KB article is now in Pinecone")
print("="*70)
