import hashlib
import json
import time
from ecdsa import SigningKey, SECP256k1
from cryptography.hazmat.primitives import serialization

# Token economy configuration
TOKEN_NAME = 'MyToken'
TOKEN_SYMBOL = 'MTK'
TOKEN_DECIMALS = 18
TOKEN_TOTAL_SUPPLY = 1000000

# Blockchain configuration
BLOCKCHAIN_NETWORK_ID = 'my_network'
BLOCKCHAIN_CHAIN_ID = 'my_chain'

# Wallet class
class Wallet:
    def __init__(self):
        self.private_key = SigningKey.from_secret_exponent(123456789, curve=SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
        self.address = self.public_key.to_string().hex()

    def sign_transaction(self, transaction):
        signature = self.private_key.sign(transaction.encode())
        return signature.hex()

    def get_address(self):
        return self.address

# Token class
class Token:
    def __init__(self, token_name, token_symbol, token_decimals, token_total_supply):
        self.token_name = token_name
        self.token_symbol = token_symbol
        self.token_decimals = token_decimals
        self.token_total_supply = token_total_supply
        self.balance = {wallet.get_address(): token_total_supply for wallet in [Wallet()]}  # Initialize balance for the creator wallet

    def transfer(self, sender_wallet, recipient_wallet, amount):
        if sender_wallet.get_address() not in self.balance:
            raise ValueError("Sender wallet not found")
        if recipient_wallet.get_address() not in self.balance:
            self.balance[recipient_wallet.get_address()] = 0
        if self.balance[sender_wallet.get_address()] < amount:
            raise ValueError("Insufficient balance")
        self.balance[sender_wallet.get_address()] -= amount
        self.balance[recipient_wallet.get_address()] += amount

    def get_balance(self, wallet):
        return self.balance.get(wallet.get_address(), 0)

# Transaction class
class Transaction:
    def __init__(self, sender_wallet, recipient_wallet, amount, token):
        self.sender_wallet = sender_wallet
        self.recipient_wallet = recipient_wallet
        self.amount = amount
        self.token = token
        self.timestamp = int(time.time())
        self.signature = sender_wallet.sign_transaction(self.to_string())

    def to_string(self):
        return json.dumps({
            'sender_wallet': self.sender_wallet.get_address(),
            'recipient_wallet': self.recipient_wallet.get_address(),
            'amount': self.amount,
            'token': self.token.token_symbol,
            'timestamp': self.timestamp
        })

    def verify_signature(self):
        return self.sender_wallet.private_key.verify(self.signature.encode(), self.to_string().encode())

# Blockchain class
class Blockchain:
    def __init__(self, network_id, chain_id):
        self.network_id = network_id
        self.chain_id = chain_id
        self.chain = []
        self.pending_transactions = []

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        if not self.pending_transactions:
            return
        block = {
            'transactions': self.pending_transactions,
            'timestamp': int(time.time()),
            'hash': self.calculate_hash()
        }
        self.chain.append(block)
        self.pending_transactions = []

    def calculate_hash(self):
        return hashlib.sha256(json.dumps(self.chain, sort_keys=True).encode()).hexdigest()

    def get_latest_block(self):
        return self.chain[-1]

# Example usage
if __name__ == "__main__":
    wallet1 = Wallet()
    wallet2 = Wallet()
    token = Token(TOKEN_NAME, TOKEN_SYMBOL, TOKEN_DECIMALS, TOKEN_TOTAL_SUPPLY)
    blockchain = Blockchain(BLOCKCHAIN_NETWORK_ID, BLOCKCHAIN_CHAIN_ID)

    transaction1 = Transaction(wallet1, wallet2, 100, token)
    blockchain.add_transaction(transaction1)
    blockchain.mine_block()

    print("Latest block:", blockchain.get_latest_block())
    print("Wallet 1 balance:", token.get_balance(wallet1))
    print("Wallet 2 balance:", token.get_balance(wallet2))
