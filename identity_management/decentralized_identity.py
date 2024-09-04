import os
import json
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from web3 import Web3
from web3.auto import w3

# Set up Web3 connection
def setup_web3():
    w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
    return w3

# Generate a new RSA key pair for identity management
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

# Create a decentralized identity (DID) using the RSA key pair
def create_did(private_key, public_key):
    did = hashlib.sha256(public_key).hexdigest()
    return did

# Create a verifiable credential (VC) using the DID and private key
def create_vc(did, private_key, attributes):
    vc = {
        "@context": "https://www.w3.org/2018/credentials/v1",
        "type": ["VerifiableCredential"],
        "credentialSubject": attributes,
        "issuer": did,
        "issuanceDate": "2023-02-20T14:30:00Z",
        "expirationDate": "2024-02-20T14:30:00Z"
    }
    signed_vc = sign_vc(vc, private_key)
    return signed_vc

# Sign a verifiable credential using the private key
def sign_vc(vc, private_key):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    signed_vc = private_key.sign(
        json.dumps(vc, separators=(',', ':')).encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signed_vc

# Verify a verifiable credential using the public key
def verify_vc(vc, public_key):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    public_key.verify(
        vc,
        json.dumps(vc, separators=(',', ':')).encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return True

# Example usage
if __name__ == "__main__":
    # Set up Web3 connection
    w3 = setup_web3()

    # Generate a new RSA key pair
    private_key, public_key = generate_key_pair()

    # Create a decentralized identity (DID)
    did = create_did(private_key, public_key)

    # Create a verifiable credential (VC)
    attributes = {"name": "John Doe", "age": 30}
    vc = create_vc(did, private_key, attributes)

    # Verify the verifiable credential
    verified = verify_vc(vc, public_key)
    print(f"Verified: {verified}")

    # Store the DID and VC on the blockchain
    # (This part is omitted for brevity, but you can use Web3 to interact with a blockchain)
