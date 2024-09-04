import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Encryption configuration
SALT = os.urandom(16)
KEY_SIZE = 32

# Key derivation function
def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

# Fernet encryption class
class FernetEncryption:
    def __init__(self, password):
        self.key = derive_key(password, SALT)
        self.fernet = Fernet(self.key)

    def encrypt(self, data):
        return self.fernet.encrypt(data.encode())

    def decrypt(self, encrypted_data):
        return self.fernet.decrypt(encrypted_data).decode()

# Example usage
if __name__ == "__main__":
    password = "my_secret_password"
    encryption = FernetEncryption(password)
    data = "Hello, World!"
    encrypted_data = encryption.encrypt(data)
    print("Encrypted data:", encrypted_data)
    decrypted_data = encryption.decrypt(encrypted_data)
    print("Decrypted data:", decrypted_data)
