import binascii

class OneTimePad:

    def is_ascii_alphabetic(self, b):
        return (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A)

    def read_file(self, filename):
        ciphertexts = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                lines = line.strip()
                decoded = binascii.unhexlify(lines)
                ciphertexts.append(decoded)
        return ciphertexts

    def xor_bytes(self, a, b):
        """XOR two byte sequences of the same length"""
        return bytes(x ^ y for x, y in zip(a, b))

    def xor_to_binary(self, xor_result):
        """Convert XOR result bytes to a space-separated binary string."""
        return " ".join(f"{byte:08b}" for byte in xor_result)


    def xor_ciphertexts(self, ciphertexts):
        """XOR every pair of ciphertexts to reveal plaintext differences"""
        for i in range(len(ciphertexts)):
            for j in range(i + 1, len(ciphertexts)):
                result = self.xor_bytes(ciphertexts[i], ciphertexts[j])
                binary_representation = self.xor_to_binary(result)
                print(f"C{i+1} ⊕ C{j+1}: {binary_representation}")

    def recover_key(self, ciphertexts):
        key = bytearray(len(ciphertexts[0]))

        for i in range(len(ciphertexts) - 2):
            for j in range(i + 1, len(ciphertexts) - 1):
                for k in range(j + 1, len(ciphertexts)):
                    self.find_key(key, ciphertexts[i], ciphertexts[j], ciphertexts[k])

        return bytes(key)

    def find_key(self, key, c1, c2, c3):
        for b in range(len(c1)):
            if key[b] != 0x00:
                continue

            c12 = self.is_ascii_alphabetic(c1[b] ^ c2[b])
            c13 = self.is_ascii_alphabetic(c1[b] ^ c3[b])
            c23 = self.is_ascii_alphabetic(c2[b] ^ c3[b])

            if c12 and c13:
                key[b] = c1[b] ^ 0x20
            elif c12 and c23:
                key[b] = c2[b] ^ 0x20
            elif c13 and c23:
                key[b] = c3[b] ^ 0x20

    def decrypt(self, key, ciphertexts):
        plaintexts = []

        for ct in ciphertexts:
            plaintext = bytes(k ^ b for k, b in zip(key, ct))
            plaintexts.append(plaintext)

        return plaintexts

    def manual_adjustments(self, key):
        key = bytearray(key)

        key[0] ^= 0xbb ^ ord('I')
        key[6] ^= ord('O') ^ ord('l')
        key[8] ^= ord('W') ^ ord('n')
        key[10] ^= 0xa7 ^ ord('i')
        key[11] ^= 0x6e ^ ord('n')
        key[17] ^= 0x85 ^ ord('e')
        key[20] ^= ord('O') ^ ord('e')
        key[29] ^= 0x80 ^ ord('n')

        return key


if __name__ == '__main__':
    onetimepad = OneTimePad()
    filename = "ciphertext.txt"

    # Read and decrypt
    ciphertexts = onetimepad.read_file(filename)
    i=0
    for ct in ciphertexts:
        i=i+1
        print("The ciphertext:", i)
        print(" ".join(f"{byte:08b}" for byte in ct))

    print("\nXOR Ciphertext Pairs:")
    onetimepad.xor_ciphertexts(ciphertexts)

    key = onetimepad.recover_key(ciphertexts)
    print("\n")
    binary_key = " ".join(f"{byte:08b}" for byte in key)
    print("The key is:", binary_key)

    plaintexts = onetimepad.decrypt(key, ciphertexts)

    # Print decrypted messages
    for i, pt in enumerate(plaintexts):
        print(f"{i + 1}) {pt.decode(errors='ignore')}")

    print("\nFinal decryptions with manual adjustments:")

    # Apply manual adjustments and decrypt again
    adjusted_key = onetimepad.manual_adjustments(key)
    plaintexts = onetimepad.decrypt(adjusted_key, ciphertexts)

    # Print final results
    for i, pt in enumerate(plaintexts):
        print(f"{i + 1}) {pt.decode(errors='ignore')}")
