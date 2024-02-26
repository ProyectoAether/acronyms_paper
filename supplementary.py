from functools import partial
import argparse
import time

import einops
import torch
from IPython.display import clear_output
# TransformerLens
from transformer_lens import HookedTransformer, patching

from dataset import *
from mechanistic_utils import *
from plotly_utils import imshow, scatter
import plotly.express as px
# My libraries
from utils import *

torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=50, help="number of samples to use")
args = parser.parse_args()

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True
)

with open("acronyms_2_common.txt", "r") as f:
   prompts, acronyms = list(zip(*[line.split(", ") for line in f.read().splitlines()]))

n = args.n
prompts = prompts[:n]
acronyms = acronyms[:n]

clean_tokens = model.to_tokens(prompts)
answer_tokens = model.to_tokens(acronyms, prepend_bos=False)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)

heads_to_visualize = [[1, 0], [2, 2], [4, 11]]

labels = ["BOS", "The", "C1", "T1", "C2", "T2", "C3", "T3", " (", "A1", "A2"]

mean_attn_patterns = torch.stack(
   [clean_cache["pattern", layer][:, head].mean(0) for layer, head in heads_to_visualize],
   dim=0)

imshow(
   mean_attn_patterns, facet_col=0, facet_labels=[f"{layer}.{head}" for layer, head in heads_to_visualize],
   title="Mean Attention Patterns for Fuzzy Previous Heads", width=800, height=400, x=labels, y=labels,
   labels={"y": "Destination", "x":"Source"}, save_path="images/supplementary/attn_patterns_fuzzy.pdf"
)