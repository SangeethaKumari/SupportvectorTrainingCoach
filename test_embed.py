# Save as test_embed.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
## to test the embedding model and its
## Intermittent API Downtime: Google's Generative AI services
#  occasionally experience transient 500 errors.
# Connecting to Qdrant at: https://38970671-a23a-48a1-a38c-3b64d16c36da.us-east4-0.gcp.cloud.qdrant.io
# Error in retrieve node: Error embedding content: 500 INTERNAL. 
# {'error': {'code': 500, 'message': 'Internal error encountered.', 'status': 'INTERNAL'}}

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector = embeddings.embed_query("This is a test.")
    print(f"✅ Success! Embedding dimension: {len(vector)}")
except Exception as e:
    print(f"❌ Failed: {str(e)}")
