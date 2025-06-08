import binascii
import math
from collections import Counter

MAX_KEY_LEN = 13

# 🔹 Sum of occurrences
def sum_arr(arr):
    return sum(arr)

# 🔹 Sum of squared frequencies
def sum_arr_squared(arr):
    return sum(math.pow(x, 2) for x in arr)

# 🔹 Average of an array
def avg_arr(arr):
    return sum(arr) / len(arr) if arr else 0

# 🔹 XOR operation
def xor_arr(arr, k):
    return bytes([b ^ k for b in arr])

# 🔹 Compute frequency distributions for key lengths 1 to MAX_KEY_LEN
def get_stream_frequencies(ct):
    vec = []
    for L in range(1, MAX_KEY_LEN + 1):
        vec.append([Counter() for _ in range(L)])

    
    for i, b in enumerate(ct):
        for L in range(1, MAX_KEY_LEN + 1):
            vec[L - 1][i % L][b] += 1

    freq = [[{} for _ in range(L)] for L in range(1, MAX_KEY_LEN + 1)]
    
    for L in range(1, MAX_KEY_LEN + 1):
        for i, blk in enumerate(vec[L - 1]):
            tot = sum_arr(blk.values())
            freq[L - 1][i] = {j: blk[j] / tot for j in blk}
    print(freq)
    return freq

# 🔹 Valid plaintext characters: letters, punctuation, spaces (no numbers)
def try_key(ct_stream, candidate_key):
    candidate_pt = xor_arr(ct_stream, candidate_key)
    
    for b in candidate_pt:
        if not (b == 0x20 or  # space
                b == 0x2C or  # ,
                b == 0x2E or  # .
                (0x41 <= b <= 0x5A) or  # A-Z
                (0x61 <= b <= 0x7A)):  # a-z
            return False
    return True

# 🔹 Find the correct key for a given ciphertext stream
def find_key(ct_stream):
    for k in range(256):
        if try_key(ct_stream, k):
            return k, True
    return 0x00, False


if __name__ == "__main__":
    filename = "ctext.txt"
    with open(filename, "r") as f:
        ciphertext_hex = f.read().strip().replace("\n", "")

    ciphertext_bytes = binascii.unhexlify(ciphertext_hex)

    # 🔹 Get the sum of squared frequencies for each key length
    freq = get_stream_frequencies(ciphertext_bytes)

    sums = []
    for L in range(1, MAX_KEY_LEN + 1):
        squared_sums = [sum_arr_squared(freq[L - 1][i].values()) for i in range(L)]
        sums.append(avg_arr(squared_sums))

    # 🔹 Find best key length
    keylen = max(range(1, MAX_KEY_LEN + 1), key=lambda i: sums[i - 1])
    print(f"Candidate key length: {keylen}")

    # 🔹 Build ciphertext streams
    ct_streams = [[] for _ in range(keylen)]
    for i, b in enumerate(ciphertext_bytes):
        ct_streams[i % keylen].append(b)
    
    # 🔹 Find key
    key = bytearray(keylen)
    for i in range(keylen):
        print(f"Searching key byte at index {i}...")
        found_key, found = find_key(ct_streams[i])
        if found:
            key[i] = found_key
        else:
            print("  ---> NOT FOUND!")

    key_hex = key.hex()
    print(f"Recovered Key: {key_hex}")

    # 🔹 Decrypt ciphertext
    plaintext = bytes([ciphertext_bytes[i] ^ key[i % keylen] for i in range(len(ciphertext_bytes))]).decode(errors="ignore")
    print("\n📜 Decrypted Text:\n", plaintext)
