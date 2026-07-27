import torch
import torch.nn as nn
from full_transformer import MAX_LENGTH, EMBEDDING_SIZE, HEAD_SIZE, NUM_DECODER_BLOCKS, NUM_HEADS, token_to_char, vocab_size, GPT, tokenizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

