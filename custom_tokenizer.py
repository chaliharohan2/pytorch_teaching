def tokenize(text: str):
    encoded_text = list(text.encode("utf-8"))
    print("Original length: ", len(encoded_text))

    merges, _ = _encode(encoded_text)

    # take the raw bytes for tokens from 0 to 255 which is range for utf-8
    vocab = {idx: bytes([idx]) for idx in range(256)}

    # add the bytes for merged tokens (which is sum of bytes of merged pair)
    for merge, val in merges.items():
        vocab[val] = vocab[merge[0]] + vocab[merge[1]]

    return merges, vocab

def _encode(encoded_text: list):
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

    return merges

def decode(tokens: list, vocab: dict):

    input_bytes = [vocab[tok] for tok in tokens]

    decoded_str = [inp.decode(encoding="utf-8", errors="replace") for inp in input_bytes]
    return decoded_str