import os
import hashlib
import hmac
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding as symmetric_padding

# Load the private key and certificate from files
with open("private_key.pem", "rb") as key_file:
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
        backend=default_backend()
    )

with open("certificate.pem", "rb") as cert_file:
    certificate = load_pem_x509_certificate(
        cert_file.read(),
        backend=default_backend()
    )

# Define the encryption and decryption functions
def encrypt_message(message, public_key):
    # Generate a random symmetric key
    symmetric_key = os.urandom(32)

    # Encrypt the symmetric key with the public key
    encrypted_symmetric_key = public_key.encrypt(
        symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Encrypt the message with the symmetric key
    cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(os.urandom(16)), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = symmetric_padding.PKCS7(128).padder()
    padded_message = padder.update(message) + padder.finalize()
    encrypted_message = encryptor.update(padded_message) + encryptor.finalize()

    # Return the encrypted symmetric key and message
    return encrypted_symmetric_key, encrypted_message

def decrypt_message(encrypted_symmetric_key, encrypted_message, private_key):
    # Decrypt the symmetric key with the private key
    symmetric_key = private_key.decrypt(
        encrypted_symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt the message with the symmetric key
    cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(encrypted_message[:16]), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_message = decryptor.update(encrypted_message[16:]) + decryptor.finalize()
    unpadder = symmetric_padding.PKCS7(128).unpadder()
    decrypted_message = unpadder.update(decrypted_padded_message) + unpadder.finalize()

    # Return the decrypted message
    return decrypted_message

# Define the digital signature functions
def sign_message(message, private_key):
    # Generate a hash of the message
    message_hash = hashlib.sha256(message).digest()

    # Sign the hash with the private key
    signature = private_key.sign(
        message_hash,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # Return the signature
    return signature

def verify_signature(message, signature, certificate):
    # Extract the public key from the certificate
    public_key = certificate.public_key()

    # Generate a hash of the message
    message_hash = hashlib.sha256(message).digest()

    # Verify the signature with the public key
    public_key.verify(
        signature,
        message_hash,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

# Define the messaging functions
def send_message(message, recipient_public_key):
    # Encrypt the message with the recipient's public key
    encrypted_symmetric_key, encrypted_message = encrypt_message(message, recipient_public_key)

    # Sign the encrypted message with the sender's private key
    signature = sign_message(encrypted_message, private_key)

    # Return the encrypted message and signature
    return encrypted_symmetric_key, encrypted_message, signature

def receive_message(encrypted_symmetric_key, encrypted_message, signature, sender_certificate):
    # Verify the signature with the sender's certificate
    verify_signature(encrypted_message, signature, sender_certificate)

    # Decrypt the symmetric key with the recipient's private key
    symmetric_key = private_key.decrypt(
        encrypted_symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt the message with the symmetric key
    decrypted_message = decrypt_message(encrypted_symmetric_key,
