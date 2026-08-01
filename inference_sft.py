import torch
import torch.nn as nn
import json
from full_transformer import MAX_LENGTH, GPT, EMBEDDING_SIZE, NUM_HEADS, HEAD_SIZE, NUM_DECODER_BLOCKS
from sft import tokenizer, new_vocab_size, token_to_char

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = GPT(vocab_size=new_vocab_size, max_context_length=MAX_LENGTH, embedding_size=EMBEDDING_SIZE, num_heads=NUM_HEADS, head_size=HEAD_SIZE, num_decoder_layers=NUM_DECODER_BLOCKS)

model.load_state_dict(torch.load("./full_transformer_tiny_shakespeare_sft.pt"))
model.eval()

print("Generate: ")

# from new line start
# model.generate(500, token_to_char=token_to_char, max_context_length=MAX_LENGTH, start_token=0)

# from sentence start
question = "Who is Gremio?"
print(question, "\n")
model.generate_assistant_response(question=question, tokenizer=tokenizer, max_tokens=500, token_to_char=token_to_char, max_context_length=MAX_LENGTH, greedy=True)
print("\n")