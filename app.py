import requests
from flask import Flask, request, Response, jsonify
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# from functions import chunk_and_store, generate_response
# from dotenv import load_dotenv
import os

# load_dotenv()

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')

print("STARTING", TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY)

genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel("models/gemini-1.5-flash-8b-latest")

def message_parser(message):
    try:
        chat_id = message['message']['chat']['id']
        if 'text' in message['message']:
            text = message['message']['text']
        else:
            text = '__NONE__'
        
        # Check if there is a document in the message
        if 'document' in message['message']:
            file_id = message['message']['document']['file_id']
        else:
            file_id = None
    except:
        chat_id = -1
        text = '__NONE__'
        file_id = None
    
    print("Chat ID: ", chat_id)
    print("Message: ", text)
    print("File ID: ", file_id)
    
    return chat_id, text, file_id

def download_document(file_id):
    file_info_url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}'
    file_info_response = requests.get(file_info_url).json()
    
    if 'result' in file_info_response:
        file_path = file_info_response['result']['file_path']
        download_url = f'https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}'
        
        # Download the file
        response = requests.get(download_url)
        
        # Save the file locally
        file_name = os.path.join('/', os.path.basename(file_path))
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print("File saved as:", file_name)
        return file_name
    return None

def send_message_telegram(chat_id, text):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode' : 'Markdown'
    }
    response = requests.post(url, json=payload)
    return response

@app.route('/hello', methods=['GET'])
def hello():
    try:
        # print(generate_response("What is your name?", llm))
        return f"{TELEGRAM_BOT_TOKEN} and {GOOGLE_API_KEY}"
    except Exception as e:
        print("OOPS SOMETHING WENT WRONG WITH GENERATE RESPONSE", e)
    return "Hello world"

# @app.route('/world', methods=['GET'])
# def world():
#     print(chunk_and_store(r"C:\Users\shaharyar\Documents\VS Code\Topics in LLMs\Project\outline.pdf"))
#     return "Hello world"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        msg = request.get_json()
        chat_id, incoming_que, file_id = message_parser(msg)
        if chat_id != -1:
            if file_id:
                # Download and save the document if it exists
                file_path = download_document(file_id)
                # chunk the file and add to mongo db
                # chunk_msg = chunk_and_store(file_path)
                chunk_msg = "oops"

                if file_path:
                    send_message_telegram(chat_id, f"Document saved successfully: {chunk_msg}")
                else:
                    send_message_telegram(chat_id, "Failed to save the document." + chunk_msg)
                try:
                    os.remove(file_path)
                    print("Deleting", file_path)
                except:
                    pass
            elif incoming_que.strip() == '/chatid':
                send_message_telegram(chat_id, f'Your chat ID is: {chat_id}')
            else:
                # answer = generate_response(incoming_que, llm)
                pass
                # send_message_telegram(chat_id, "Echo: " + incoming_que)
        return Response('ok', status=200)
    else:
        return "<h1>GET Request Made</h1>"

if __name__ == '__main__':
    app.run()