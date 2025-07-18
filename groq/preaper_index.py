from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

# Optional: Wipe old index
if os.path.exists("faiss_index"):
    import shutil
    shutil.rmtree("faiss_index")
    print("Old FAISS index removed.")

# Step 1: Load medical text (ensure this file exists)
loader = TextLoader("medical_knowledge.txt")
docs = loader.load()

# Step 2: Split documents for embedding
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
final_documents = splitter.split_documents(docs)

# Step 3: Generate embeddings using Ollama + nous-hermes2 (ensure Ollama is running)
embeddings = OllamaEmbeddings(model="nous-hermes2")
faiss_index = FAISS.from_documents(final_documents, embeddings)

# Step 4: Save FAISS index
faiss_index.save_local("faiss_index")
print("✅ FAISS index with medical knowledge saved!")
