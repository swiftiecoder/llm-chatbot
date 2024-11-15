import hashlib
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
# rag chain imports
from langchain.prompts import PromptTemplate
# create stuff document chain imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema.runnable import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from uuid import uuid4
import os

load_dotenv()

embedding_dim = 768
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')

# Initialize Pinecone
pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "llms-project"

# custom_rag_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=(
#         "Use the following context to answer the question.\n\n"
#         "Context:\n{context}\n\n"
#         "Question:\n{question}\n\n"
#         "Provide a detailed and helpful response:"
#     )
# )

template = """
Use the following context to answer the question.

Context: {context}
Question: {question}
Answer: 

"""

custom_rag_prompt = PromptTemplate(
  template=template, 
  input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# format_docs = RunnableLambda(
#     lambda docs: "\n".join([f"Document {i + 1}: {doc.page_content}" for i, doc in enumerate(docs)])
#     if isinstance(docs, list) and all(isinstance(doc, Document) for doc in docs)
#     else ValueError("Input to format_docs must be a list of Document objects.")
# )

def initialize_vector_store():
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
        pass
    pinecone_index = pinecone.Index(index_name)

    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, 
            model="models/text-embedding-004"
        )
    )
    return vector_store


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

def generate_response_with_rag(query, llm):
    try:
        vector_store = initialize_vector_store()
        retriever = vector_store.as_retriever(
            # search_type="mmr",
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5},
        )
        # print vector store
        print(vector_store)
    
        
        # print("query:", query)  
        rag_chain = (
            {"context": retriever | format_docs,  "question": RunnablePassthrough()} 
            | custom_rag_prompt 
            | llm
            | StrOutputParser() 
        )
        
    
        result = rag_chain.invoke(query)
        return result 
    except Exception as e:
        print(f"Error in generate_response_with_rag: {e}")
        return f"Error in generate_response_with_rag: {e}"

