import os
import json
import ipfshttpclient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

# Set up IPFS connection
def setup_ipfs():
    client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
    return client

# Generate a new RSA key pair for encryption
def generate_key_pair():
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    private_key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key = key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    )
    return private_key, public_key

# Encrypt a file using the RSA key pair
def encrypt_file(file_path, private_key, public_key):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    encrypted_data = encrypt_data(file_data, public_key)
    return encrypted_data

# Encrypt data using the RSA public key
def encrypt_data(data, public_key):
    from cryptography.hazmat.primitives.asymmetric import padding
    encrypted_data = public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_data

# Upload a file to IPFS
def upload_file_to_ipfs(client, file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    res = client.add(file_data)
    return res

# Pin a file to IPFS
def pin_file_to_ipfs(client, cid):
    client.pin.add(cid)

# Example usage
if __name__ == "__main__":
    # Set up IPFS connection
    client = setup_ipfs()

    # Generate a new RSA key pair
    private_key, public_key = generate_key_pair()

    # Encrypt a file
    file_path = "example.txt"
    encrypted_data = encrypt_file(file_path, private_key, public_key)

    # Upload the encrypted file to IPFS
    res = upload_file_to_ipfs(client, encrypted_data)
    cid = res['Hash']

    # Pin the file to IPFS
    pin_file_to_ipfs(client, cid)

    print(f"File uploaded to IPFS with CID: {cid}")
