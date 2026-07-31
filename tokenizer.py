text = "In response to this incident, we began a large-scale retrospective review of our own cybersecurity evaluations. In particular, we looked for evidence that Claude—like the OpenAI models that accessed Hugging Face—was able to access the internet from within testing environments that should have been sealed off. 😀 😃 😄 😁 🥳"
encoded_text = list(text.encode("utf-8"))
# print(encoded_text)

pair_counts = {}
for pair in zip(encoded_text[:-1], encoded_text[1:]):
    pair_counts[pair] = pair_counts.get(pair, 0) + 1
# print(pair_counts)

top_pair = max(pair_counts, key=pair_counts.get)
print(top_pair, f" / \"{chr(top_pair[0])} {chr(top_pair[1])}\": ", pair_counts[top_pair])