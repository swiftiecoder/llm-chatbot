from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
# from dotenv import load_dotenv
from uuid import uuid4
import os

# load_dotenv()

# import nltk
# from nltk.corpus import stopwords

# Ensure stopwords are downloaded
# nltk.download('stopwords')

embedding_dim = 512
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')

def generate_response(query, model):
    # print("In generate content")
    try:
        response = model.generate_content(query)
        # print(response.text)
        return response.text
    except Exception as e:
        # print("An exception occured")
        return e


def chunk_and_store(file_path):
    try:
        # Load the document using PyMuPDF
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        # Initialize text splitter and split documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)

        # Remove stopwords from text chunks
        # stop_words = set(stopwords.words("english"))
        processed_chunks = []
        for chunk in chunks:
            # words = chunk.split()
            # filtered_text = " ".join([word for word in words if word.lower() not in stop_words])
            # processed_chunks.append(filtered_text)
            processed_chunks.append(chunk)

        # Generate embeddings with Google model
        genai_embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, 
            model="models/text-embedding-004", 
            dimension=embedding_dim
        )

        # Initialize MongoDB client and vector store
        client = MongoClient(MONGO_DB_URI)
        db = client["llms-project-db"]
        collection = db["vectorstore"]
        
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=genai_embeddings,
            index_name="index-vectorstore",
            relevance_score_fn="cosine"
        )

        # Create vector search index with specified dimensions
        vector_store.create_vector_search_index(dimensions=embedding_dim)

        # Insert documents with embeddings into the vector store
        # print("chunks", chunks)

        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids = uuids)


        return "Documents successfully embedded and stored in MongoDB."
    except Exception as e:
        print("Exception thrown")
        return e
