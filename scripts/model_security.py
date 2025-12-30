
import os
import glob
from cryptography.fernet import Fernet
import argparse
import sys

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
EXTENSIONS_TO_ENCRYPT = ['*.pkl', '*.json']

def generate_key():
    """Generates a key and prints it."""
    key = Fernet.generate_key()
    print(f"GENERATED KEY: {key.decode()}")
    print("Store this key safely! Add it to your GitHub Secrets as 'MODEL_ENCRYPTION_KEY'.")
    return key

def get_key(args):
    """Retrieves key from args or environment variable."""
    if args.key:
        return args.key.encode()
    
    env_key = os.environ.get('MODEL_ENCRYPTION_KEY')
    if env_key:
        return env_key.encode()
    
    print("Error: No encryption key provided. Use --key or set MODEL_ENCRYPTION_KEY environment variable.")
    sys.exit(1)

def encrypt_files(key):
    """Encrypts matching files in the models directory."""
    f = Fernet(key)
    count = 0
    files_to_encrypt = []
    
    for ext in EXTENSIONS_TO_ENCRYPT:
        files_to_encrypt.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
    
    for file_path in files_to_encrypt:
        # Skip if already encrypted extension or if it is the key itself (unlikely)
        if file_path.endswith('.enc'):
            continue
            
        print(f"Encrypting {os.path.basename(file_path)}...")
        
        with open(file_path, 'rb') as file:
            file_data = file.read()
        
        encrypted_data = f.encrypt(file_data)
        
        enc_path = file_path + '.enc'
        with open(enc_path, 'wb') as file:
            file.write(encrypted_data)
            
        count += 1
        
    print(f"Encrypted {count} files.")

def decrypt_files(key):
    """Decrypts .enc files in the models directory."""
    f = Fernet(key)
    count = 0
    encrypted_files = glob.glob(os.path.join(MODELS_DIR, '*.enc'))
    
    for enc_path in encrypted_files:
        # Determine original filename (remove .enc)
        original_path = enc_path[:-4]
        print(f"Decrypting {os.path.basename(enc_path)}...")
        
        with open(enc_path, 'rb') as file:
            encrypted_data = file.read()
        
        try:
            decrypted_data = f.decrypt(encrypted_data)
        except Exception as e:
            print(f"Failed to decrypt {enc_path}: {e}")
            continue
            
        with open(original_path, 'wb') as file:
            file.write(decrypted_data)
            
        count += 1
        
    print(f"Decrypted {count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encrypt/Decrypt proprietary model files.')
    parser.add_argument('action', choices=['encrypt', 'decrypt', 'generate-key'], help='Action to perform')
    parser.add_argument('--key', help='Encryption key (optional if MODEL_ENCRYPTION_KEY env var is set)')
    
    args = parser.parse_args()
    
    if args.action == 'generate-key':
        generate_key()
    elif args.action == 'encrypt':
        key = get_key(args)
        encrypt_files(key)
    elif args.action == 'decrypt':
        key = get_key(args)
        decrypt_files(key)
