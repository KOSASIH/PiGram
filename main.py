import logging
from logging import getLogger
from blockchain import Blockchain
from chatbot import Chatbot
from utils.logging import get_logger
from utils.settings import *

# Initialize the logger
logger = get_logger()
logger.info("Application started")

# Initialize the blockchain
blockchain = Blockchain(BLOCKCHAIN_NETWORK, BLOCKCHAIN_NODE_URL)
logger.info("Blockchain initialized")

# Initialize the chatbot
chatbot = Chatbot(CHATBOT_WELCOME_MESSAGE, CHATBOT_HELP_MESSAGE)
logger.info("Chatbot initialized")

def main():
    try:
        # Start the chatbot
        chatbot.start()
        logger.info("Chatbot started")

        # Start the blockchain node
        blockchain.start_node()
        logger.info("Blockchain node started")

        # Run the application
        while True:
            # Handle incoming messages
            message = chatbot.get_message()
            if message:
                logger.info(f"Received message: {message}")
                response = chatbot.handle_message(message)
                logger.info(f"Responding with: {response}")
                chatbot.send_message(response)

            # Mine a new block every 10 seconds
            blockchain.mine_block("Mining data")
            logger.info("Mined a new block")

    except KeyboardInterrupt:
        logger.info("Application stopped")
        chatbot.stop()
        blockchain.stop_node()

if __name__ == "__main__":
    main()
