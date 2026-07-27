import torch
import torch.nn as nn
from full_transformer import MAX_LENGTH, EMBEDDING_SIZE, HEAD_SIZE, NUM_DECODER_BLOCKS, NUM_HEADS, token_to_char, vocab_size, GPT, tokenizer
import json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = GPT(vocab_size=vocab_size, max_context_length=MAX_LENGTH, embedding_size=EMBEDDING_SIZE, num_heads=NUM_HEADS, head_size=HEAD_SIZE, num_decoder_layers=NUM_DECODER_BLOCKS)

model.load_state_dict(torch.load("./full_transformer_tiny_shakespeare.pt"))

# update the tokenizer with new tokens for user message start, assistant message start, end of sequence, and padding tokens
new_tokens = ["<user>", "<assistant>", "<end>", "<pad>"]
for tok in new_tokens:
    if tok not in tokenizer:
        token_to_char[len(tokenizer)] = tok
        tokenizer[tok] = len(tokenizer)

new_vocab_size = len(tokenizer)

def update_embeddings(old_text_embeddings: nn.Embedding, new_vocab_size: int):
    new_text_embeddings = nn.Embedding(new_vocab_size, EMBEDDING_SIZE, device=device)
    with torch.no_grad():
        new_text_embeddings.weight[:old_text_embeddings.weight.size(0), :] = old_text_embeddings.weight
    return new_text_embeddings

def update_final_proj_layer(old_linear_proj: nn.Linear, new_vocab_size: int):
    new_linear_output = nn.Linear(EMBEDDING_SIZE, new_vocab_size, device=device)

    with torch.no_grad():
        new_linear_output.weight[:old_linear_proj.weight.size(0), :] = old_linear_proj.weight
        new_linear_output.bias[:old_linear_proj.bias.size(0)] = old_linear_proj.bias

    return new_linear_output
    
if __name__ == "__main__":
    model.text_embedding_table = update_embeddings(old_text_embeddings=model.text_embedding_table, new_vocab_size=new_vocab_size)
    model.linear_output = update_final_proj_layer(old_linear_proj=model.linear_output, new_vocab_size=new_vocab_size)