import requests
from flask import Flask, request, Response, jsonify
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os

app = Flask(__name__)

telegram_bot_token = os.environ.get('BOT_TOKEN')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro')
chats = {}
last_prompt = 'Empty'

def chat_length(chat_id):
    if chat_id in chats.keys():
        return len(chats[chat_id].history)//2
    else:
        return -1
        
def chat_exists(chat_id):
    if chat_id in chats.keys():
        return True
    else:
        return False
        
def create_chat(chat_id):
    if chat_id == -1:
        return
    if chat_exists(chat_id):
        return
    else:
        chats[chat_id] = model.start_chat(history=[])

def generate_answer(chat_id, question):
    try:
        if not chat_exists(chat_id):
            create_chat(chat_id)
        response = chats[chat_id].send_message(question, safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        })
        print(response)
        return response.text
    except:
        return "Something went wrong generating the response"

def message_parser(message):
    try:
        chat_id = message['message']['chat']['id']
        try:
            text = message['message']['text']
        except:
            text = '__NONE__'
    except:
        chat_id = -1
        text = '__NONE__'
    print("Chat ID: ", chat_id)
    print("Message: ", text)
    return chat_id, text

def send_message_telegram(chat_id, text):
    url = f'https://api.telegram.org/bot{telegram_bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode' : 'Markdown'
    }
    response = requests.post(url, json=payload)
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        msg = request.get_json()
        chat_id, incoming_que = message_parser(msg)
        if chat_id != -1:
            if incoming_que.strip() == '/chatid':
                send_message_telegram(chat_id, f'Your chat ID is: {chat_id}')
            elif incoming_que.strip() == '/chatlen':
                send_message_telegram(chat_id, f'Your chat length is: {chat_length(chat_id)}')
            elif incoming_que.strip() == '/start':
                create_chat(chat_id)
                start_msg = "Hi there!"
                send_message_telegram(chat_id, start_msg)
            elif incoming_que.strip() == '/numusers':
                send_message_telegram(chat_id, f'There are {len(chats)} users')
            elif incoming_que.strip() == '/removeme':
                del chats[chat_id]
                send_message_telegram(chat_id, 'You have been removed')
            elif incoming_que.strip() == '/removeall':
                chats.clear()
            elif incoming_que.strip() == '/lastprompt':
                send_message_telegram(chat_id, last_prompt)
            elif incoming_que.strip() == '/chathistory':
                try:
                    print(chats[chat_id].history)
                    send_message_telegram(chat_id, chats[chat_id].history)
                except:
                    send_message_telegram(chat_id, "Something went wrong")
            elif incoming_que.strip() == '/chatdic':
                try:
                    print(chats)
                    send_message_telegram(chat_id, chats)
                except:
                    send_message_telegram(chat_id, "Something went wrong")
            elif incoming_que == '__NONE__':
                send_message_telegram(chat_id, 'Sorry, I can only interact with text right now :(')
            else:
                answer = generate_answer(chat_id, incoming_que)
                send_message_telegram(chat_id, answer)
        return Response('ok', status=200)
    else:
        # print(telegram_bot_token, GOOGLE_API_KEY)
        return "<h1>GET Request Made</h1>"


if __name__ == '__main__':
    app.run()
