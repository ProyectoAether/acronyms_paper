from functools import partial
import argparse
import time

import torch
from IPython.display import clear_output
# TransformerLens
from transformer_lens import HookedTransformer, patching

import plotly.express as px

from dataset import *
from mechanistic_utils import *
from plotly_utils import imshow
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

corrupted_tokens = clean_tokens.clone()
corrupted_tokens = corrupted_tokens[torch.randperm(corrupted_tokens.shape[0])]

corrupted_tokens_acronym = clean_tokens.clone()
corrupted_tokens_acronym = corrupted_tokens_acronym[torch.randperm(corrupted_tokens_acronym.shape[0])]

corrupted_tokens[:, -2:] = corrupted_tokens_acronym[:, -2:]

corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)


def get_attn_df(layer, head, cache):
    attn_patterns = cache[utils.get_act_name("pattern", layer)] # (batch, n_heads, q_pos, k_pos)
    attn_patterns = attn_patterns[:, head] # (batch_size, q_pos, k_pos)
    # Take the attention paid from the END, A1 and A2 tokens
    attn_patterns = attn_patterns[:, 8:11, 1:] # Discard the BOS Token
    # Compute the mean and std
    mean_attn, std_attn = attn_patterns.mean(0), attn_patterns.std(0)
    labels = ["BOS", "The", "C1", "T1", "C2", "T2", "C3", "T3", " (", "A1", "A2"]
    attn_df = pd.DataFrame()
    color = ["The", "C", "T", "C", "T", "C", "T", "A", "A", "A"]
    #color = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 2, 2]
    attn_df["Attention Probability"] = mean_attn.reshape(-1).cpu()
    attn_df["std"] = std_attn.reshape(-1).cpu()
    attn_df["Token"] = labels[1:] * 3
    attn_df["Color"] = color * 3
    attn_df["Letter"] = torch.tensor([1, 2, 3]).repeat_interleave(10)

    return attn_df


layer = 8
head = 11
attn_df = get_attn_df(layer, head, clean_cache)

fig = px.bar(attn_df, x="Token", y="Attention Probability", error_y="std", facet_col="Letter",
       title=f"Avg. Attention paid at each prediction by head {layer}.{head}", color="Color", category_orders={'Token': attn_df.Token},
       width=1000)

fig.update_layout(showlegend=False)
fig.write_image(f"images/histograms/{layer}_{head}.pdf")
time.sleep(1)
fig.write_image(f"images/histograms/{layer}_{head}.pdf")