import requests
from flask import Flask, request, Response, jsonify
from functions import generate_response, generate_response_with_rag, chunk_and_store, manage_chat_history, get_chat_history
from dotenv import load_dotenv
import os
import threading

load_dotenv()

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

# print("STARTING", TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY)

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

def process_document_async(file_path, chat_id):
    try:
        # Process and store document chunks
        chunk_msg = chunk_and_store(file_path, chat_id)
        send_message_telegram(chat_id, f"Document processed successfully: {chunk_msg}")
    except Exception as e:
        send_message_telegram(chat_id, f"Error processing document: {e}")
    finally:
        # Attempt to delete the file 
        try:
            os.remove(file_path)
            print("Deleting", file_path)
        except Exception as e:
            send_message_telegram(chat_id, f"Error deleting file: {e}")

def send_message_telegram(chat_id, text):
    manage_chat_history(chat_id, text, 'bot')
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
    # print("In hello")
    try:
        return generate_response_with_rag("Who are you?", 7194910082)
        # return generate_response("what is your name", llm)
        # return f"{TELEGRAM_BOT_TOKEN} and {GOOGLE_API_KEY}"
    except Exception as e:
        print("OOPS SOMETHING WENT WRONG WITH GENERATE RESPONSE", e)
    return "Hello world"

    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            msg = request.get_json()
            chat_id, incoming_que, file_id = message_parser(msg)
        except Exception as e:
            error_msg = f"Error parsing message: {e}"
            if 'chat_id' in locals():
                send_message_telegram(chat_id, error_msg)
            return Response('Failed to parse message', status=400)
        
        if chat_id != -1:
            if incoming_que.strip() and not file_id:
                try: 
                    manage_chat_history(chat_id, incoming_que, "user")
                except Exception as e:
                    send_message_telegram(chat_id, f"Error managing chat history: {e}")
                    return Response('Failed to manage chat history', status=500)
                
            if file_id:
                # Try downloading the document
                try:
                    file_path = download_document(file_id)
                    if file_path:
                        # Start an asynchronous thread to process the document
                        threading.Thread(
                            target=process_document_async, 
                            args=(file_path, chat_id), 
                            daemon=True
                        ).start()
                        
                        # Immediately return a 200 OK response
                        send_message_telegram(chat_id, "Please wait while the document is being processed")
                    else:
                        send_message_telegram(chat_id, "Failed to save the document.")
                except Exception as e:
                    send_message_telegram(chat_id, f"Error handling document download: {e}")

            elif incoming_que.strip() == '/chatid':
                try:
                    send_message_telegram(chat_id, f'Your chat ID is: {chat_id}')
                except Exception as e:
                    send_message_telegram(chat_id, f"Error sending chat ID: {e}")
                    return Response('Failed to send chat ID', status=500)
            else:
                try:
                    # answer = generate_response(incoming_que, llm)
                    answer = generate_response_with_rag(incoming_que, chat_id)
                    send_message_telegram(chat_id, answer)
                except Exception as e:
                    send_message_telegram(chat_id, f"Error generating or sending response: {e}")
                    return Response('Failed to send response', status=500)
        
        return Response('ok', status=200)
    else:
        return "<h1>GET Request Made</h1>"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)