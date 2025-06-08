#!/usr/bin/env python

import sys
import binascii


# Extended English letter frequency including spaces and punctuation
englishLetterFreq = {
    ' ': 17.00, 'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
    'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76,
    'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
    'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07, 
    ',': 1.10, '.': 0.60, ';': 0.10, ':': 0.09, '!': 0.05, '?': 0.06
}

ETAOIN = ' ETAOINSHRDLCUMWFGYPBVKJXQZ'  # Include space at the beginning
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '


# Return the number of different bits between two 32-bit numbers
def dif(a, b, bits=32):
    x = a ^ b
    return sum((x >> i) & 1 for i in range(bits))

# Compute the edit/Hamming distance between two byte sequences (of equal lengths)
# Returns number of differing bits
def d_ham(s1, s2):
    return sum(dif(s1[i], s2[i]) for i in range(len(s1)))

# Used below for single-char xor decoding (step f.)
def transpose(blocks):
    t = []
    for i in range(len(blocks[0])):
        t1 = []
        for j in range(len(blocks)):
            t1.append(blocks[j][i])
        t.append(bytes(t1))  # Ensure we return bytes, not strings
    return t

charset = "abcdefghijklmnopqrstuvwxyz "
charset_up = charset.upper()
punct = ",.;?!:"

# Simple scoring method based on character frequency
def get_score(text):
    # Score: number of valid characters
    s = 0
    for l in text.decode(errors="ignore"):  # Decode bytes to string safely
        if (l in charset) or (l in charset_up) or (l in punct):
            s += 1
    return (s * 1.0) / len(text) if len(text) > 0 else 0

# Decrypt/encrypt function (same function for both)
def rxor(text, key):
    dec = bytes([text[i] ^ key[i % len(key)] for i in range(len(text))])
    return dec

if __name__ == "__main__":

    fname = "ctext.txt"
    with open(fname, "r") as f:
        data = f.read().strip()
    enc = binascii.unhexlify(data)

    # Find the most probable key sizes
    d = {}
    for keysize in range(2, 13):
        blocks = [enc[i:i+keysize] for i in range(0, len(enc), keysize)]
        dist = 0
        # Average Hamming distance over 5 blocks
        for i in range(0, 5):
            for j in range(i+1, 5):
                dist += d_ham(blocks[i], blocks[j]) * 1.0 / keysize
        d[keysize] = dist / 10

    print("Most probable key sizes: ")
    for w in sorted(d, key=lambda k: d[k]):
        print("Key size:", w, "Hamming dist:", d[w])
        # Most possible key sizes: 2, 3, 5. Test them in turn
    
    # Split into KEYSIZE blocks and crack individually
    for keysize in [7]:  # Try different key sizes
        k = [0] * keysize

        blocks = [enc[i:i+keysize] for i in range(0, len(enc), keysize)]
        # Transpose the blocks and find the key for each transposed block
        # Leave the last incomplete block for easier calculations
        t_blocks = transpose(blocks[:len(blocks)-1])

        for i in range(keysize):
            b = t_blocks[i]

            d = {}
            for key in range(0, 255):
                dec = bytes([c ^ key for c in b])
                d[dec] = (get_score(dec), key, b)

            print("Possible keys for transposed block %d. keysize: %d" % (i, keysize))
         
            for w in sorted(d, key=lambda k: d[k][0], reverse=True)[:1]:
                print(d[w][1], "score:", d[w][0])
                k[i] = d[w][1]

        k = bytes(k)
        print("Key:", binascii.hexlify(k).decode())
        print("Message:", rxor(enc, k).decode(errors="ignore")) 
