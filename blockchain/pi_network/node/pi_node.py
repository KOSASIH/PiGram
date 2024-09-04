import os
import sys
import logging
import asyncio
import json
from typing import Dict, List
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from pi_network import PiNetwork
from pi_wallet import PiWallet
from pi_blockchain import PiBlockchain
from pi_transaction import PiTransaction
from pi_block import PiBlock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PiNode:
    def __init__(self, node_id: str, private_key: str, public_key: str, node_address: str, node_port: int):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = public_key
        self.node_address = node_address
        self.node_port = node_port
        self.pi_network = PiNetwork()
        self.pi_wallet = PiWallet()
        self.pi_blockchain = PiBlockchain()
        self.pi_transaction = PiTransaction()
        self.pi_block = PiBlock()

    async def start_node(self):
        # Start the node and connect to the Pi Network
        logger.info(f"Starting node {self.node_id}...")
        await self.pi_network.connect_to_network(self.node_address, self.node_port)
        logger.info(f"Node {self.node_id} connected to the Pi Network.")

    async def stop_node(self):
        # Stop the node and disconnect from the Pi Network
        logger.info(f"Stopping node {self.node_id}...")
        await self.pi_network.disconnect_from_network()
        logger.info(f"Node {self.node_id} disconnected from the Pi Network.")

    async def create_transaction(self, sender_address: str, recipient_address: str, amount: float):
        # Create a new transaction
        logger.info(f"Creating transaction from {sender_address} to {recipient_address}...")
        transaction = self.pi_transaction.create_transaction(sender_address, recipient_address, amount)
        logger.info(f"Transaction created: {transaction}")

    async def broadcast_transaction(self, transaction: Dict):
        # Broadcast the transaction to the Pi Network
        logger.info(f"Broadcasting transaction {transaction['transaction_id']}...")
        await self.pi_network.broadcast_transaction(transaction)
        logger.info(f"Transaction {transaction['transaction_id']} broadcasted.")

    async def mine_block(self):
        # Mine a new block
        logger.info(f"Mining new block...")
        block = self.pi_block.mine_block(self.pi_blockchain.get_latest_block())
        logger.info(f"Block mined: {block}")

    async def add_block_to_blockchain(self, block: Dict):
        # Add the block to the blockchain
        logger.info(f"Adding block {block['block_id']} to the blockchain...")
        self.pi_blockchain.add_block(block)
        logger.info(f"Block {block['block_id']} added to the blockchain.")

    async def get_blockchain(self):
        # Get the current state of the blockchain
        logger.info(f"Getting blockchain...")
        blockchain = self.pi_blockchain.get_blockchain()
        logger.info(f"Blockchain: {blockchain}")

    async def get_node_info(self):
        # Get information about the node
        logger.info(f"Getting node info...")
        node_info = {
            "node_id": self.node_id,
            "node_address": self.node_address,
            "node_port": self.node_port,
            "public_key": self.public_key
        }
        logger.info(f"Node info: {node_info}")

async def main():
    # Create a new node
    node = PiNode(
        node_id="node_1",
        private_key="private_key_1",
        public_key="public_key_1",
        node_address="localhost",
        node_port=8080
    )

    # Start the node
    await node.start_node()

    # Create a new transaction
    await node.create_transaction("sender_address", "recipient_address", 10.0)

    # Broadcast the transaction
    await node.broadcast_transaction({
        "transaction_id": "transaction_1",
        "sender_address": "sender_address",
        "recipient_address": "recipient_address",
        "amount": 10.0
    })

    # Mine a new block
    await node.mine_block()

    # Add the block to the blockchain
    await node.add_block_to_blockchain({
        "block_id": "block_1",
        "transactions": [
            {
                "transaction_id": "transaction_1",
                "sender_address": "sender_address",
                "recipient_address": "recipient_address",
                "amount": 10.0
            }
        ]
    })

    # Get the current state of the blockchain
    await node.get_blockchain()

    # Get information about the node
    await node.get_node_info()

    # Stop the node
    await node.stop_node()

if __name__ == "__main__":
    asyncio.run(main())
