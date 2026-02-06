import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

def ingest_pdfs(data_dir="./data"):
    # 1. Initialize Qdrant Client 
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("COLLECTION_NAME", "llm_course_material")

    # Auto-adjust data_dir if it's not found or empty (e.g. if running from inside backend/ folder)
    # We check if the current data_dir has PDFs, if not, we check one level up.
    has_pdfs = os.path.exists(data_dir) and any(f.endswith(".pdf") for f in os.listdir(data_dir))
    
    if not has_pdfs and os.path.exists("../data"):
        if any(f.endswith(".pdf") for f in os.listdir("../data")):
            data_dir = "../data"
            print(f"📂 Found PDF files in {data_dir}")

    if qdrant_url and "localhost" not in qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        connection_args = {"url": qdrant_url, "api_key": qdrant_api_key}
    else:
        client = QdrantClient(path="./qdrant_db")
        connection_args = {"path": "./qdrant_db"}
    
    # 2. Setup Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 3. Load and Split Documents
    all_docs = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            print(f"📄 Processing {file}...")
            loader = PyMuPDFLoader(os.path.join(data_dir, file))
            docs = loader.load()
            all_docs.extend(docs)
            
    if not all_docs:
        print(f"❌ No PDF files found in {data_dir} directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(all_docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    # 4. Create collection if not exists
    if not client.collection_exists(collection_name):
        print(f"🛠 Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
    # Close the client before VectorStore opens it
    client.close()

    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        batch_size=64, # Smaller batches for more reliable cloud uploading
        **connection_args
    )
    print("✨ Ingestion complete!")

if __name__ == "__main__":
    ingest_pdfs()
