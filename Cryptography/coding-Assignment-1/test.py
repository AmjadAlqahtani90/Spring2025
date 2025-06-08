import binascii

def vigenere_encrypt(plaintext, key):
    key_bytes = key.encode()
    plaintext_bytes = plaintext.encode()
    ciphertext = bytes([plaintext_bytes[i] ^ key_bytes[i % len(key)] for i in range(len(plaintext_bytes))])
    return binascii.hexlify(ciphertext).decode()

plaintext = """Cryptography is the practice and study of techniques for secure communication."""
key = "securekey"  # Example key

ciphertext_hex = vigenere_encrypt(plaintext, key)

# Save to file
with open("test_ctext.txt", "w") as f:
    f.write(ciphertext_hex)

print(f"Ciphertext saved to test_ctext.txt: {ciphertext_hex}")
