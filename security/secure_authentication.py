import os
import hashlib
import hmac
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Authentication configuration
RSA_KEY_SIZE = 2048
HASH_ALGORITHM = hashes.SHA256()

# RSA key pair generation
def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=RSA_KEY_SIZE,
    )
    public_key = private_key.public_key()
    return private_key, public_key

# Digital signature class
class DigitalSignature:
    def __init__(self, private_key):
        self.private_key = private_key

    def sign(self, data):
        signature = self.private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(HASH_ALGORITHM),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            HASH_ALGORITHM,
        )
        return signature

    def verify(self, public_key, data, signature):
        public_key.verify(
            signature,
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(HASH_ALGORITHM),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            HASH_ALGORITHM,
        )

# Example usage
if __name__ == "__main__":
    private_key, public_key = generate_key_pair()
    digital_signature = DigitalSignature(private_key)
    data = "Hello, World!"
    signature = digital_signature.sign(data)
    print("Signature:", signature)
    digital_signature.verify(public_key, data, signature)
    print("Verification successful!")
