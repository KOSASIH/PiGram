import unittest
from chatbot import Chatbot

class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.chatbot = Chatbot()

    def test_greet_user(self):
        response = self.chatbot.handle_message("/start")
        self.assertEqual(response, "Welcome to our chatbot!")

    def test_handle_help(self):
        response = self.chatbot.handle_message("/help")
        self.assertEqual(response, "This is a help message.")

    def test_handle_unknown_message(self):
        response = self.chatbot.handle_message("Unknown message")
        self.assertEqual(response, "I didn't understand that. Try /help for more information.")

    def test_conversation_flow(self):
        responses = []
        responses.append(self.chatbot.handle_message("/start"))
        responses.append(self.chatbot.handle_message("/help"))
        responses.append(self.chatbot.handle_message("Unknown message"))
        self.assertEqual(responses, ["Welcome to our chatbot!", "This is a help message.", "I didn't understand that. Try /help for more information."])

if __name__ == "__main__":
    unittest.main()
