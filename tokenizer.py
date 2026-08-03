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
        vocab[val] = bytes([merge[0]]) + bytes([merge[1]])

    input_bytes = [vocab[tok] for tok in tokens]

    decoded_str = [inp.decode() for inp in input_bytes]
    return decoded_str

    

text = "In response to this incident, we began a large-scale retrospective review of our own cybersecurity evaluations. In particular, we looked for evidence that Claude—like the OpenAI models that accessed Hugging Face—was able to access the internet from within testing environments that should have been sealed off. 😀 😃 😄 😁 🥳"
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

