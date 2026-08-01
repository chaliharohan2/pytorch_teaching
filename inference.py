import torch
import torch.nn as nn
import json
from full_transformer import MAX_LENGTH, EMBEDDING_SIZE, HEAD_SIZE, NUM_DECODER_BLOCKS, NUM_HEADS, token_to_char, vocab_size, GPT, tokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = GPT(vocab_size=vocab_size, max_context_length=MAX_LENGTH, embedding_size=EMBEDDING_SIZE, num_heads=NUM_HEADS, head_size=HEAD_SIZE, num_decoder_layers=NUM_DECODER_BLOCKS)

model.load_state_dict(torch.load("./full_transformer_tiny_shakespeare.pt"))
model.eval()

print("Generate: ")

# from new line start
# model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH, start_token=0)

# from sentence start
start_word = "JULIET:"
print(start_word, end="", flush=True)
model.generate(800, token_to_char=token_to_char, max_context_length=MAX_LENGTH, start_token=[tokenizer.get(char) for char in start_word])
print("\n")