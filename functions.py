from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
# import nltk
# from nltk.corpus import stopwords

# Ensure stopwords are downloaded
# nltk.download('stopwords')

embedding_dim = 512
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')

def generate_response(query, model):
    response = model.generate_content("Write a story about a magic backpack.")
    return response


def chunk_and_store(genai, file_path):
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
    embeddings = []
    for text in processed_chunks:
        embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(embedding['embedding'])

    # Initialize MongoDB client and vector store
    client = MongoClient(MONGO_DB_URI)
    db = client["llms-project-db"]
    collection = db["vectorstore"]
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=Embeddings(dimension=embedding_dim),  # Specify the dimension
        index_name="index-vectorstores",
        relevance_score_fn="cosine"
    )

    # Create vector search index with specified dimensions
    vector_store.create_vector_search_index(dimensions=embedding_dim)

    # Insert documents with embeddings into the vector store
    for i, text in enumerate(processed_chunks):
        vector_store.add_document(id=f"doc_{i}", content=text, embedding=embeddings[i])

    return "Documents successfully embedded and stored in MongoDB."

# Example usage
# chunk_and_store(genai, "/path/to/document.pdf", "your_mongodb_atlas_uri")