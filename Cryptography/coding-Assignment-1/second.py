import string
from collections import Counter

# XOR-encrypted ciphertext (Hex)
ciphertext_hex = "F96DE8C227A259C87EE1DA2AED57C93FE5DA36ED4EC87EF2C63AAE5B9A7EFFD673BE4ACF7BE8923CAB1ECE7AF2DA3DA44FCF7AE29235A24C963FF0DF3CA3599A70E5DA36BF1ECE77F8DC34BE129A6CF4D126BF5B9A7CFEDF3EB850D37CF0C63AA2509A76FF9227A55B9A6FE3D720A850D97AB1DD35ED5FCE6BF0D138A84CC931B1F121B44ECE70F6C032BD56C33FF9D320ED5CDF7AFF9226BE5BDE3FF7DD21ED56CF71F5C036A94D963FF8D473A351CE3FE5DA3CB84DDB71F5C17FED51DC3FE8D732BF4D963FF3C727ED4AC87EF5DB27A451D47EFD9230BF47CA6BFEC12ABE4ADF72E29224A84CDF3FF5D720A459D47AF59232A35A9A7AE7D33FB85FCE7AF5923AA31EDB3FF7D33ABF52C33FF0D673A551D93FFCD33DA35BC831B1F43CBF1EDF67F0DF23A15B963FE5DA36ED68D378F4DC36BF5B9A7AFFD121B44ECE76FEDC73BE5DD27AFCD773BA5FC93FE5DA3CB859D26BB1C63CED5CDF3FE2D730B84CDF3FF7DD21ED5ADF7CF0D636BE1EDB79E5D721ED57CE3FE6D320ED57D469F4DC27A85A963FF3C727ED49DF3FFFDD24ED55D470E69E73AC50DE3FE5DA3ABE1EDF67F4C030A44DDF3FF5D73EA250C96BE3D327A84D963FE5DA32B91ED36BB1D132A31ED87AB1D021A255DF71B1C436BF479A7AF0C13AA14794"

# Convert hex to bytes
ciphertext = bytes.fromhex(ciphertext_hex)
cipher_len = len(ciphertext)

# Expanded English letter frequency table (Lowercase, Uppercase, Spaces, Punctuation)
frequency_table = {
    # Lowercase letters
    ord('a'): 8.12, ord('b'): 1.49, ord('c'): 2.71, ord('d'): 4.32, ord('e'): 12.02,
    ord('f'): 2.30, ord('g'): 2.03, ord('h'): 5.92, ord('i'): 7.31, ord('j'): 0.10,
    ord('k'): 0.69, ord('l'): 3.98, ord('m'): 2.61, ord('n'): 6.96, ord('o'): 7.68,
    ord('p'): 1.82, ord('q'): 0.11, ord('r'): 6.02, ord('s'): 6.28, ord('t'): 9.10,
    ord('u'): 2.88, ord('v'): 1.11, ord('w'): 2.09, ord('x'): 0.17, ord('y'): 2.11,
    ord('z'): 0.07,
    
    # Uppercase letters (same frequency as lowercase)
    ord('A'): 8.12, ord('B'): 1.49, ord('C'): 2.71, ord('D'): 4.32, ord('E'): 12.02,
    ord('F'): 2.30, ord('G'): 2.03, ord('H'): 5.92, ord('I'): 7.31, ord('J'): 0.10,
    ord('K'): 0.69, ord('L'): 3.98, ord('M'): 2.61, ord('N'): 6.96, ord('O'): 7.68,
    ord('P'): 1.82, ord('Q'): 0.11, ord('R'): 6.02, ord('S'): 6.28, ord('T'): 9.10,
    ord('U'): 2.88, ord('V'): 1.11, ord('W'): 2.09, ord('X'): 0.17, ord('Y'): 2.11,
    ord('Z'): 0.07,

    # Spaces and punctuation
    ord(' '): 13.00, ord('.'): 1.10, ord(','): 1.10, ord('?'): 0.05, ord('!'): 0.05,
    ord(';'): 0.05, ord(':'): 0.05, ord('\''): 0.05, ord('"'): 0.05, ord('-'): 0.05,
    ord('('): 0.05, ord(')'): 0.05, ord('['): 0.05, ord(']'): 0.05, ord('/'): 0.05,
    
    # Numbers (optional)
    ord('0'): 1.00, ord('1'): 1.00, ord('2'): 1.00, ord('3'): 1.00, ord('4'): 1.00,
    ord('5'): 1.00, ord('6'): 1.00, ord('7'): 1.00, ord('8'): 1.00, ord('9'): 1.00
}

# Identified key length from step 1
key_length = 7

# Step 1: Group ciphertext bytes by key position
key_bytes = [[] for _ in range(key_length)]
for i in range(cipher_len):
    key_bytes[i % key_length].append(ciphertext[i])

# Step 2: Recover each key byte using frequency analysis
recovered_key = []

for pos in range(key_length):
    best_score = float('-inf')
    best_key_byte = 0

    for trial_key in range(256):
        # XOR each byte in the column with the candidate key byte
        decoded_text = [c ^ trial_key for c in key_bytes[pos]]
        
        # Compute frequency score based on expanded frequency table
        score = sum(frequency_table.get(c, 0) for c in decoded_text)

        # Select key byte with the highest score
        if score > best_score:
            best_score = score
            best_key_byte = trial_key

    recovered_key.append(best_key_byte)

# Print recovered key
print("Recovered Key:", bytes(recovered_key).decode('latin1'))

# Step 3: Decrypt the full ciphertext using the recovered key
def decrypt(ciphertext, key):
    key_length = len(key)
    plaintext = bytes(ciphertext[i] ^ key[i % key_length] for i in range(cipher_len))
    return plaintext.decode('latin1', errors='replace')  # Replace unprintable characters

decrypted_text = decrypt(ciphertext, recovered_key)
print("\nDecrypted Text:")
print(decrypted_text)
