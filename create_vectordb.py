import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import ast

load_dotenv()

# 🔗 Your data sources
links_list = ["https://www.rguktrkv.ac.in/Departments.php?view=EC&staff=TS" , "https://www.rguktrkv.ac.in/Departments.php?view=EC&staff=NTS" , "https://www.rguktrkv.ac.in/Syllabus.php?view=ECE" , "https://www.rguktrkv.ac.in/#"] 
print("🔄 Loading web data...")
web_loader = WebBaseLoader(links_list)
webdata = web_loader.load()

print("🔄 Loading PDF data...")
pdf_loader = PyPDFLoader("pics&pdf/ece_syllabus_mini.pdf")
pdf_pages = [doc for doc in pdf_loader.lazy_load()]

# Combine all data
all_docs = webdata + pdf_pages

print("✂️ Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(all_docs)

print("🧠 Creating embeddings...")
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

print("📦 Creating FAISS DB...")
db = FAISS.from_documents(chunks, embeddings)

# 💾 Save locally
db.save_local("faiss_index")

print("✅ FAISS index saved successfully!")