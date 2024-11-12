import hashlib
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
from uuid import uuid4
import os

# load_dotenv()

embedding_dim = 768
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')

# Initialize Pinecone
pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "llms-project"

def generate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):  # Read file in chunks to avoid memory issues
            sha256.update(chunk)
    return sha256.hexdigest()

# def file_already_uploaded(ids, vector_store):
#     # Query Pinecone or a metadata DB for this file path
#     try:
#         #print("Checking if file already exists")
#         result = vector_store.get_by_ids(ids)
#         return len(result) > 0
#     except Exception as e:
#         print("Error checking file existence:", e)
#         return False

def generate_response(query, model):
    try:
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return str(e)

def chunk_and_store(file_path):
    try:
        #print("Starting")
        # Initialize Pinecone index and vector store
        try:
            pinecone.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1")
            )
        except:
            #print("Index already exists")
            pass
        pinecone_index = pinecone.Index(index_name)

        #print("Made index if doesn't exist")

        vector_store = PineconeVectorStore(
            index=pinecone_index,
            embedding=GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY, 
                model="models/text-embedding-004"
            )
        )

        #print("made it here")

        file_hash = generate_file_hash(file_path)

        # Load and split the document
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)

        # Insert document chunks with unique IDs and file path metadata
        uuids = [str(file_hash)+str(i) for i in range(len(chunks))]
        
        # Check if file has already been processed and stored
        # if file_already_uploaded(uuids, vector_store):
        #     #print("File already uploaded; reuploading")
        #     pass

        vector_store.add_documents(documents=chunks, ids=uuids)

        return "Documents successfully embedded and stored in Pinecone."
    except Exception as e:
        #print("Exception thrown:", e)
        return str(e)
