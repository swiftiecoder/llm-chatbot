from flask import Flask
import os

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')

@app.route('/')
def hello():
    return f"{TELEGRAM_BOT_TOKEN} and {GOOGLE_API_KEY}"

if __name__ == '__main__':
    app.run()