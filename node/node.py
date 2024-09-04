import os
import socket
import pickle
import threading
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Node configuration
NODE_ID = 'node_1'
NODE_PORT = 8080
NODE_HOST = 'localhost'
NODE_KEY = b'your_secret_key_here'

# Cryptography setup
kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    length=32,
    salt=b'your_salt_here',
    iterations=100000,
)
key = base64.urlsafe_b64encode(kdf.derive(NODE_KEY))
fernet = Fernet(key)

# Node class
class Node:
    def __init__(self, node_id, node_port, node_host):
        self.node_id = node_id
        self.node_port = node_port
        self.node_host = node_host
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((node_host, node_port))
        self.socket.listen(5)
        self.peers = {}

    def start(self):
        print(f"Node {self.node_id} started on {self.node_host}:{self.node_port}")
        while True:
            conn, addr = self.socket.accept()
            print(f"Connected by {addr}")
            threading.Thread(target=self.handle_connection, args=(conn,)).start()

    def handle_connection(self, conn):
        while True:
            data = conn.recv(1024)
            if not data:
                break
            decrypted_data = fernet.decrypt(data)
            print(f"Received: {decrypted_data.decode()}")
            # Process the received data here
            response = b"Response from node"
            encrypted_response = fernet.encrypt(response)
            conn.sendall(encrypted_response)
        conn.close()

    def send_data(self, peer_id, data):
        peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        peer_socket.connect((self.peers[peer_id][0], self.peers[peer_id][1]))
        encrypted_data = fernet.encrypt(data.encode())
        peer_socket.sendall(encrypted_data)
        peer_socket.close()

    def add_peer(self, peer_id, peer_host, peer_port):
        self.peers[peer_id] = (peer_host, peer_port)

    def remove_peer(self, peer_id):
        del self.peers[peer_id]

# Example usage
if __name__ == "__main__":
    node = Node(NODE_ID, NODE_PORT, NODE_HOST)
    node.start()
