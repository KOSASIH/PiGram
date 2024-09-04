import requests
import json
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Telegram API configuration
TELEGRAM_API_TOKEN = "YOUR_TELEGRAM_API_TOKEN"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}"

# Telegram API class
class TelegramAPI:
    def __init__(self, api_token):
        self.api_token = api_token
        self.api_url = f"https://api.telegram.org/bot{api_token}"

    def send_message(self, chat_id, message):
        data = {"chat_id": chat_id, "text": message}
        response = requests.post(f"{self.api_url}/sendMessage", json=data)
        return response.json()

    def get_updates(self, offset=0):
        data = {"offset": offset}
        response = requests.post(f"{self.api_url}/getUpdates", json=data)
        return response.json()

    def handle_update(self, update):
        if "message" in update:
            message = update["message"]
            chat_id = message["chat"]["id"]
            text = message["text"]
            if text == "/start":
                self.send_message(chat_id, "Welcome to our Telegram bot!")
            elif text == "/help":
                self.send_message(chat_id, "This is a help message.")
            else:
                self.send_message(chat_id, "I didn't understand that. Try /help for more information.")

# Telegram bot class
class TelegramBot:
    def __init__(self, api_token):
        self.api_token = api_token
        self.telegram_api = TelegramAPI(api_token)
        self.updater = Updater(api_token, use_context=True)

    def start_polling(self):
        self.updater.start_polling()
        self.updater.idle()

    def handle_message(self, update, context):
        self.telegram_api.handle_update(update)

# Example usage
if __name__ == "__main__":
    telegram_api_token = "YOUR_TELEGRAM_API_TOKEN"
    telegram_bot = TelegramBot(telegram_api_token)
    telegram_bot.start_polling()
