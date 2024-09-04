import os
import socket
import pyaudio
import wave
import hashlib
import hmac
import base64
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate

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

# Define the audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

# Define the encryption and decryption functions
def encrypt_audio(audio, public_key):
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

    # Encrypt the audio with the symmetric key
    cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(os.urandom(16)), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_audio = encryptor.update(audio) + encryptor.finalize()

    # Return the encrypted symmetric key and audio
    return encrypted_symmetric_key, encrypted_audio

def decrypt_audio(encrypted_symmetric_key, encrypted_audio, private_key):
    # Decrypt the symmetric key with the private key
    symmetric_key = private_key.decrypt(
        encrypted_symmetric_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Decrypt the audio with the symmetric key
    cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(encrypted_audio[:16]), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_audio = decryptor.update(encrypted_audio[16:]) + decryptor.finalize()

    # Return the decrypted audio
    return decrypted_audio

# Define the digital signature functions
def sign_audio(audio, private_key):
    # Generate a hash of the audio
    audio_hash = hashlib.sha256(audio).digest()

    # Sign the hash with the private key
    signature = private_key.sign(
        audio_hash,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # Return the signature
    return signature

def verify_signature(audio, signature, certificate):
    # Extract the public key from the certificate
    public_key = certificate.public_key()

    # Generate a hash of the audio
    audio_hash = hashlib.sha256(audio).digest()

    # Verify the signature with the public key
    public_key.verify(
        signature,
        audio_hash,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

# Define the voice call functions
def make_call(recipient_ip, recipient_port):
    # Create a socket for the voice call
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((recipient_ip, recipient_port))

    # Open the audio stream
    stream = pyaudio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)

    # Load the recipient's public key
    with open("recipient_public_key.pem", "rb") as key_file:
        recipient_public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )

    # Send the audio data
    while True:
        audio_data = stream.read(CHUNK)
        encrypted_symmetric_key, encrypted_audio = encrypt_audio(audio_data, recipient_public_key)
        signature = sign_audio(encrypted_audio, private_key)
        sock.sendall(encrypted_symmetric_key + encrypted_audio + signature)

    # Close the socket and audio stream
    sock.close()
    stream.stop_stream()
    stream.close()

def receive_call(listen_ip, listen_port):
    # Create a socket for the voice call
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((listen_ip, listen_port))
    sock.listen(1)

    # Accept the incoming connection
    conn, addr = sock.accept()

    # Open the audio stream
    stream = pyaudio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          output=True,
                          frames_per_buffer=CHUNK)

    # Receive the audio data
    while True:
        data = conn.recv(1024)
        if not data:
            break
        encrypted_symmetric_key = data[:256]
        encrypted_audio = data[256:-256]
        signature = data[-256:]
        decrypted_audio = decrypt_audio(encrypted_symmetric_key, encrypted_audio, private_key)
        verify_signature(decrypted_audio, signature, certificate)
        stream.write(decrypted_audio)

    # Close the socket and audio stream
    conn.close()
    stream.stop_stream()
    stream.close()

# Example usage
if __name__ == "__main__":
    # Make a voice call
    make_call("192.168.1.100", 12345)

    # Receive a voice call
    receive_call("192.168.1.100", 12345)
