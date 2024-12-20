import hashlib
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# rag chain imports
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

from datetime import datetime, timezone

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import json
from duckduckgo_search import DDGS

load_dotenv()

embedding_dim = 768
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
MONGO_DB_HISTORY = os.environ.get('MONGO_DB_HISTORY')

genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest",
                            system_instruction="""
                            You are a chatbot designed to answer in the style of a character from a book within a messaging app. 
                            If the character is provided to you, answer as them, otherwise you are a chatbot named Portal-LLM. 
                            Talk like a normal person if no character is provided to you.
                            """,
                            generation_config={
                                "response_mime_type": "text/plain"}
                            )

classifier = genai.GenerativeModel("gemini-1.5-flash-8b-latest",
                                #    system_instruction="You are a classifier whose job is to extract the subject of a sentence, if there is one.",
                                   generation_config={"response_mime_type": "application/json"})

# Connect to MongoDB
client = MongoClient(MONGO_DB_URI)
db = client['langchain-db']
history_db = db['history']
characters_db = db['character']


def handle_character_request(chat_id, query):
    # Step 1: Classify the query
    classification = classify(query)

    print("Classification", classification)

    if (classification == None) or (classification and "none" in classification.lower()):
        character = characters_db.find_one({'chat_id': chat_id})

        if character and 'last_character' in character:
            return character['last_character']
        else:
            return ""
    else:
        characters_db.update_one(
            {'chat_id': chat_id},
            {'$set': {'last_character': classification}},
            upsert=True
        )
        return classification


# Initialize Pinecone
pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Add 'Answer like'
template = """
You are {character}. Act and talk like {character} in your responses.
If needed, you may use the provided web results, chat history and context from the book to answer the question.
Think step by step before producing a response.

<Chat History>{chat_history}</Chat History>
<Web Results>{web_results}</Web Results>
<Context>{context}</Context>
<Question>{question}</Question>

"""

custom_rag_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question", "web_results", "chat_history", "character"]
)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def get_web_results(query):
    print("Searching the web for", query)
    try:
        results = DDGS().chat(query)  # , model='claude-3-haiku')
    except:
        results = DDGS().answers(query)

    # print("The results are", results)
    return results


def classify(query):
    print("Attempting to classify:", query)
    try:
        prompt = f"""
        Rules:
        1. If the query is an instruction to act like or become a specific character/persona, return that character/persona as the subject.
        2. If the query does not contain a clear instruction to become someone else, return 'None'.
        3. Ignore queries about identity or general questions.

        Examples:
        - "Act like Harry Potter" -> "Harry Potter"
        - "Pretend to be Sherlock Holmes" -> "Sherlock Holmes"
        - "Who are you?" -> "None"
        - "Tell me about yourself" -> "None"

        <Query>{query}</Query>

        """

        print(prompt)
        print(classifier.generate_content(prompt).text)

        # answer = json.loads(classifier.generate_content(prompt).text)
        # answer =  str(next(iter(answer.values())))
        
        answer = classifier.generate_content(prompt).text
        return answer
    except:
        pass


def initialize_vector_store(chat_id):
    try:
        pinecone.create_index(
            name=str(chat_id),
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1")
        )
    except:
        pass
    pinecone_index = pinecone.Index(str(chat_id))

    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY,
            model="models/text-embedding-004",
            task_type="clustering"
        )
    )
    return vector_store


def generate_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_response(query):
    try:
        response = llm.generate_content(query.text)
        print(response.text)
        return response.text
    except Exception as e:
        print("Something went wrong in generate_response")
        return str(e)


def chunk_and_store(file_path, chat_id):
    try:
        # Initialize Pinecone index and vector store
        vector_store = initialize_vector_store(chat_id)

        file_hash = generate_file_hash(file_path)

        # Load and split the document
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=300)
        chunks = text_splitter.split_documents(documents)

        # Insert document chunks with unique IDs and file path metadata
        uuids = [str(file_hash)+str(i) for i in range(len(chunks))]

        vector_store.add_documents(documents=chunks, ids=uuids)

        return "Documents successfully embedded and stored in Pinecone."
    except Exception as e:
        # print("Exception thrown:", e)
        return str(e)


def generate_response_with_rag(query, chat_id):
    try:
        vector_store = initialize_vector_store(chat_id)
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.5},
        )

        # print(vector_store)

        character = handle_character_request(chat_id, query)
        print("character after mongodb", character)
        if character == "":
            return "You haven't asked me to act like a character yet. Please choose one"

        # Get chat history with configurable size
        chat_history = get_chat_history(chat_id) if chat_id else "No previous conversation."

        search_term = query if character is "None" else character + " " + query

        print(chat_history)

        rag_chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough(), 
             "web_results": RunnableLambda(lambda x: get_web_results(search_term)),
             "chat_history": RunnableLambda(lambda _: chat_history),
             "character": RunnableLambda(lambda _: character),
             }
            | custom_rag_prompt
            | RunnableLambda(lambda x: generate_response(x))
            # | StrOutputParser
        )

        prompt = custom_rag_prompt.format(
            context = format_docs(retriever.invoke(query)[:30]),
            question = query, 
            web_results = get_web_results(query),
            chat_history = chat_history,
            character =  character,
        )

        result = rag_chain.invoke(query)

        print(prompt)
        print(result)
        return result

    except Exception as e:
        print(f"Error in generate_response_with_rag: {e}")
        return f"Error in generate_response_with_rag: {e}"

def manage_chat_history(chat_id, message, message_type):
    # returns: A list of all messages in the session or a status message.
    try:
        # Test collection read
        test_query = history_db.find_one()
        print("Sample document:", test_query)

        # Create the chat message structure
        chat_message = {
            "timestamp": datetime.now(timezone.utc),  # Add timestamp
            "type": message_type,
            "content": message,
        }

        # Use upsert to add messages to the chat session
        history_db.update_one(
            {"chat_id": chat_id},  # Find the document by chat_id
            # Push the new message to the 'messages' array
            {"$push": {"messages": chat_message}},
            upsert=True  # Create the document if it doesn't exist
        )

        print("collection updated successfully")
    except Exception as e:
        print("Error:", e)


def get_chat_history(chat_id, k=5):
    try:
        # Get last k messages from chat history
        chat_session = history_db.find_one(
            {"chat_id": chat_id},
            {"messages": {"$slice": -(k*2)}}  # Get last k messages
        )

        if not chat_session or "messages" not in chat_session:
            return []

        # Format messages for the prompt
        formatted_history = []
        for msg in chat_session["messages"]:
            role = "User" if msg["type"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")

        return "\n".join(formatted_history)

    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return "Error retrieving chat history."
