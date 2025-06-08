package main

import (
	"encoding/hex"
	"fmt"
	"math"
)

const MAX_KEY_LEN int = 13

// number of occurrences of each char in a given block of text
type BlockCounters [256]uint
type BlockFrequencies [256]float64

// a stream (for key of len L) is an array of L block counters
// occurences at position 0,...,(L-1)
type StreamCounters []BlockCounters
type StreamFrequencies []BlockFrequencies

func sumArr(arr []uint) uint {
	var res uint = 0
	for _, x := range arr {
		res += x
	}
	return res
}

func sumArrSquared(arr []float64) float64 {
	var res float64 = 0.0
	for _, x := range arr {
		res += math.Pow(x, 2)
	}
	return res
}

func avgArr(arr []float64) float64 {
	var sum float64 = 0.0
	for _, x := range arr {
		sum += x
	}
	return sum / float64(len(arr))
}

func xorArr(arr []byte, k byte) []byte {
	res := make([]byte, len(arr))
	for i, b := range arr {
		res[i] = b ^ k
	}
	return res
}

// We start with an array of StreamCounters, one for every possible
// key lenght L=1,...,MAX_KEY_LEN.
// Then we count the occurences of each char in each stream
// and get the frequency
func getStreamFrequencies(ct []byte) [MAX_KEY_LEN]StreamFrequencies {
	var vec [MAX_KEY_LEN]StreamCounters
	for L := 1; L <= MAX_KEY_LEN; L++ {
		vec[L-1] = make([]BlockCounters, L)
	}
	for i, b := range ct {
		// increment the counter for byte 'b' in every stream
		for L := 1; L <= MAX_KEY_LEN; L++ {
			vec[L-1][i%L][b]++
		}

	}
	var freq [MAX_KEY_LEN]StreamFrequencies
	for L := 1; L <= MAX_KEY_LEN; L++ {
		freq[L-1] = make([]BlockFrequencies, L)
		for i, blk := range vec[L-1] {
			tot := sumArr(blk[:])
			for j := 0; j < len(blk); j++ {
				freq[L-1][i][j] = float64(vec[L-1][i][j]) / float64(tot)
			}
		}

	}
	return freq
}

// valid plaintext can only be:
// upper- and lower-case letters, punctuation, and spaces, but no numbers
func tryKey(ct_stream []byte, candidate_key byte) bool {
	candidate_pt := xorArr(ct_stream, candidate_key)
	for _, b := range candidate_pt {
		if !(b == 0x20 || // space
			b == 0x2C || // ,
			b == 0x2E || // .
			(b >= 0x41 && b <= 0x5A) || // upper chars
			(b >= 0x61 && b <= 0x7A)) { // lower chars
			return false
		}
	}
	return true
}

func findKey(ct_stream []byte) (byte, bool) {
	for k := 0; k <= 255; k++ {
		if tryKey(ct_stream, byte(k)) {
			return byte(k), true
		}
	}
	return 0x00, false
}

func main() {
	hex_ciphertext := "2F0C08000E1015061C5D15010C09145D02075D04101455135D0B095D060D14060C5D10151500005D1C131515055D0307150700015D"

	ct, err := hex.DecodeString(hex_ciphertext)
	if err != nil {
		panic(err)
	}

	// Get the sum of the frquencies squared for each possible key len
	freq := getStreamFrequencies(ct)
	var sums [MAX_KEY_LEN]float64
	for L := 1; L <= MAX_KEY_LEN; L++ {
		var s = make([]float64, L)
		for i := 0; i < L; i++ {
			s[i] = sumArrSquared(freq[L-1][i][:])
		}
		sums[L-1] = avgArr(s)
	}

	// Find best value --> candidate key len
	var keylen uint = 0
	var max_sum float64 = 0.0
	for i, sum := range sums {
		if sum > max_sum {
			keylen = uint(i + 1)
			max_sum = sum
		}
	}

	fmt.Printf("Candidate key len is %d\n", keylen)

	// Build ciphertext streams
	var ct_streams [][]byte = make([][]byte, keylen)
	for i, b := range ct {
		ct_streams[i%int(keylen)] = append(ct_streams[i%int(keylen)], b)
	}

	// Find key
	key := make([]byte, keylen)
	for i := 0; i < int(keylen); i++ {
		fmt.Printf("Searching key (byte at index %d)...\n", i)
		var found bool
		key[i], found = findKey(ct_streams[i])
		if !found {
			fmt.Println("  ---> NOT FOUND!")
		}
	}

	// Decrypt
	fmt.Println("Decrypting...")
	plaintext := make([]byte, len(ct))
	for i := 0; i < len(ct); i++ {
		plaintext[i] = ct[i] ^ key[i%int(keylen)]
	}

	// Result:
	/* Cryptography is the practice and study of techniques for, among other things, secure communication in the
	   presence of attackers. Cryptography has been used for hundreds, if not thousands, of years, but traditional
	   cryptosystems were designed and evaluated in a fairly ad hoc manner. For example, the Vigenere encryption
	   scheme was thought to be secure for decades after it was invented, but we now know, and this exercise
	   demonstrates, that it can be broken very easily.
	*/
	fmt.Println(string(plaintext))
}