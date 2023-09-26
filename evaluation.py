import torch

# TransformerLens
from transformer_lens import HookedTransformer

# My libraries
from utils import *
from mechanistic_utils import *
from dataset import *

torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading the model...")
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True
)
print("Loading the data...")
with open("acronyms_2_common.txt", "r") as f:
   prompts, acronyms = list(zip(*[line.split(", ") for line in f.read().splitlines()]))

n = 100
prompts = prompts[:n]
acronyms = acronyms[:n]

clean_tokens = model.to_tokens(prompts)
answer_tokens = model.to_tokens(acronyms, prepend_bos=False)
clean_logits = model(clean_tokens)
clean_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens)
print(f"Average Logit Diff. (n={n}):\n\t{clean_logit_diff.item():.2f}")