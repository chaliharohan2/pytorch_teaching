def encode(encoded_text: list):
    tokens = list(encoded_text)
    merges = {}
    j = 256
    while True:
        pair_counts = {}
        for pair in zip(tokens[:-1], tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        top_pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[top_pair] < 2:
            break

        i = 0
        while i < len(tokens)-1:
            if tokens[i] == top_pair[0] and tokens[i+1] == top_pair[1]:
                tokens[i] = j
                tokens.pop(i+1)
                merges[top_pair] = j
            i += 1
        j += 1

    return merges, tokens

def decode(tokens: list, merges: dict):
    # take the raw bytes for tokens from 0 to 255 which is range for utf-8
    vocab = {idx: bytes([idx]) for idx in range(256)}

    # add the bytes for merged tokens (which is sum of bytes of merged pair)
    for merge, val in merges.items():
        vocab[val] = vocab[merge[0]] + vocab[merge[1]]

    input_bytes = [vocab[tok] for tok in tokens]

    decoded_str = [inp.decode(encoding="utf-8", errors="replace") for inp in input_bytes]
    return decoded_str

    
with open(file="/home/rohan/Nyalazone/pytorch_testing/shakespeare_dataset.txt", mode="r") as f:
    text = f.read()
# text = "In response to this incident, we began a large-scale retrospective review of our own cybersecurity evaluations. In particular, we looked for evidence that Claude—like the OpenAI models that accessed Hugging Face—was able to access the internet from within testing environments that should have been sealed off."
# text = "Hello world, some apple in town."
encoded_text = list(text.encode("utf-8"))
print("Original length: ", len(encoded_text))

pair_counts = {}
for pair in zip(encoded_text[:-1], encoded_text[1:]):
    pair_counts[pair] = pair_counts.get(pair, 0) + 1
# print(pair_counts)

top_pair = max(pair_counts, key=pair_counts.get)
# print(top_pair, f" / \"{chr(top_pair[0])} {chr(top_pair[1])}\": ", pair_counts[top_pair])

merges, tokens = encode(encoded_text)
print(merges)
print(tokens)  
print("After tokenizing length: ", len(tokens))     

# tokens_to_decode = [276, 114, 261, 112, 277, 280, 281, 259, 262, 99, 283, 116, 285, 267, 103, 97, 258, 97, 32, 286, 103, 101, 45, 115, 99, 269, 287, 288, 289, 115, 112, 101, 99, 290, 118, 287, 270, 105, 101, 119, 291, 263, 292, 263, 119, 258, 99, 121, 267, 114, 271, 99, 292, 105, 116, 121, 293, 269, 117, 264, 105, 277, 46, 32, 276, 112, 97, 114, 290, 99, 117, 286, 285, 108, 111, 111, 107, 101, 272, 102, 111, 114, 293, 283, 99, 256, 274, 67, 268, 117, 294, 296, 108, 105, 107, 256, 297, 79, 112, 260, 65, 73, 32, 109, 111, 294, 108, 298, 300, 271, 272, 72, 117, 103, 103, 302, 70, 275, 101, 296, 119, 97, 259, 97, 98, 108, 280, 300, 259, 297, 262, 116, 101, 114, 110, 288, 32, 102, 289, 109, 266, 105, 281, 258, 116, 261, 116, 302, 260, 118, 105, 114, 265, 109, 260, 116, 298, 115, 104, 111, 117, 108, 272, 104, 97, 118, 256, 267, 101, 258, 271, 269, 101, 100, 291, 102, 46]
# print("Decoding: ", new_text := "".join(decode(tokens=tokens_to_decode, merges=merges)))
# print("Original text: ", text)
# print(text == new_text)