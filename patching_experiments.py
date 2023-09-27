from functools import partial
import argparse

import einops
import torch
from IPython.display import clear_output
# TransformerLens
from transformer_lens import HookedTransformer, patching

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

facet_labels = ["First Letter", "Second Letter", "Third Letter"]
labels = ["BOS", "The", "C1", "T1", "C2", "T2", "C3", "T3", " (", "A1", "A2"]

###############################
# 1 - CORRUPTING CURRENT WORD #
###############################

act_patch_resid_pre_iter = []
indices = [2, 4, 6]
for i, j in enumerate(indices):
    # Corrupt just the current word
    corrupted_tokens_i = clean_tokens.clone()
    corrupted_tokens_i[:, j:j+2] = corrupted_tokens[:, j:j+2] 
    _, corrupted_cache_i = model.run_with_cache(corrupted_tokens_i)

    compute_logit_diff_aux = partial(compute_logit_diff_2, answer_tokens=answer_tokens, average=False) # returns (batch_size, 3)
    compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits)[:, i].mean()
    act_patch_resid_pre = patching.get_act_patch_resid_pre(model, clean_tokens, corrupted_cache_i, compute_logit_diff_iter)
    act_patch_resid_pre_iter.append(act_patch_resid_pre)
act_patch_resid_pre_iter = torch.stack(act_patch_resid_pre_iter, dim=0)

baseline_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

fig = imshow(
    act_patch_resid_pre_iter - baseline_logit_diff[..., None, None],
    x=labels,
    facet_col=0, # This argument tells plotly which dimension to split into separate plots
    facet_col_wrap=3,
    facet_labels=facet_labels, # Subtitles of separate plots
    title="Residual Stream Patching", 
    labels={"x": "Sequence Position", "y": "Layer"},
    height=400,
    width=1000, save_path="images/activation_patching/res_1.pdf"
)

act_patch_attn_head_out_all_pos_iter = []
indices = [2, 4, 6]
for i, j in enumerate(indices):
    # Corrupt just the current word
    corrupted_tokens_i = clean_tokens.clone()
    corrupted_tokens_i[:, j:j+2] = corrupted_tokens[:, j:j+2] 
    _, corrupted_cache_i = model.run_with_cache(corrupted_tokens_i)

    compute_logit_diff_aux = partial(compute_logit_diff_2, answer_tokens=answer_tokens, average=False) # returns (batch_size, 3)
    compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits)[:, i].mean()
    act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(model, clean_tokens, corrupted_cache_i, compute_logit_diff_iter)
    act_patch_attn_head_out_all_pos_iter.append(act_patch_attn_head_out_all_pos)
act_patch_attn_head_out_all_pos_iter = torch.stack(act_patch_attn_head_out_all_pos_iter, dim=0)

baseline_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

imshow(
    act_patch_attn_head_out_all_pos_iter - baseline_logit_diff[..., None, None],
    facet_col=0,
    facet_labels=facet_labels,
    labels={"y": "Layer", "x": "Head"}, 
    title="Patching Attention Heads",
    width=800, height=400, save_path="images/activation_patching/attn_1.pdf"
)

#################################
# 2 - CORRUPTING PREVIOUS WORDS #
#################################

act_patch_resid_pre_iter = []
indices = [2, 4, 6]
for i, j in enumerate(indices):
    # Corrupt just the current word
    corrupted_tokens_i = clean_tokens.clone()
    corrupted_tokens_i[:, :j] = corrupted_tokens[:, :j] 
    _, corrupted_cache_i = model.run_with_cache(corrupted_tokens_i)
    compute_logit_diff_aux = partial(compute_logit_diff_2, answer_tokens=answer_tokens, average=False) # returns (batch_size, 3)
    compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits)[:, i].mean()
    act_patch_resid_pre = patching.get_act_patch_resid_pre(model, clean_tokens, corrupted_cache_i, compute_logit_diff_iter)
    act_patch_resid_pre_iter.append(act_patch_resid_pre)
act_patch_resid_pre_iter = torch.stack(act_patch_resid_pre_iter, dim=0)

baseline_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

imshow(
    act_patch_resid_pre_iter - baseline_logit_diff[..., None, None],
    x=labels,
    facet_col=0, # This argument tells plotly which dimension to split into separate plots
    facet_col_wrap=3,
    facet_labels=facet_labels, # Subtitles of separate plots
    title="Residual Stream Patching", 
    labels={"x": "Sequence Position", "y": "Layer"},
    height=400,
    width=1000, save_path="images/activation_patching/res_2.pdf"
)

act_patch_attn_head_out_by_pos_iter = []
indices = [2, 4, 6]
for i, j in enumerate(indices):
    # Corrupt just the current word
    corrupted_tokens_i = clean_tokens.clone()
    corrupted_tokens_i[:, :j] = corrupted_tokens[:, :j] 
    _, corrupted_cache_i = model.run_with_cache(corrupted_tokens_i)

    compute_logit_diff_aux = partial(compute_logit_diff_2, answer_tokens=answer_tokens, average=False) # returns (batch_size, 3)
    compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits)[:, i].mean()
    act_patch_attn_head_out_by_pos = patching.get_act_patch_attn_head_out_by_pos(model, clean_tokens, corrupted_cache_i, compute_logit_diff_iter)
    act_patch_attn_head_out_by_pos_iter.append(act_patch_attn_head_out_by_pos)
act_patch_attn_head_out_by_pos_iter = torch.stack(act_patch_attn_head_out_by_pos_iter, dim=0)

act_patch_attn_head_out_by_pos_iter # (3, n_layers, pos, n_heads)

act_patch_attn_head_out_by_pos_iter = einops.rearrange(act_patch_attn_head_out_by_pos_iter,
    "i n_layers pos n_heads -> i (n_layers n_heads) pos")

# Ci positions
pos_idx = [4, 6]

baseline_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

imshow(
    act_patch_attn_head_out_by_pos_iter[[1, 2], :, pos_idx].reshape(2, model.cfg.n_layers, model.cfg.n_heads) - baseline_logit_diff[1:, None, None],
    title=f"Attn. Head Patching on Position Ci", 
    labels={"x": "Head", "y": "Layer"}, facet_col=0, facet_labels=["Second Digit", "Third Digit"],
    height=400,
    width=700, save_path="images/activation_patching/attn_2_Ci.pdf"
)

# A(i-1) positions
pos_idx = [9, 10]

imshow(
    act_patch_attn_head_out_by_pos_iter[[1, 2], :, pos_idx].reshape(2, model.cfg.n_layers, model.cfg.n_heads) - baseline_logit_diff[1:, None, None],
    title=f"Attn. Head Patching on Position A(i-1)", 
    labels={"x": "Head", "y": "Layer"}, facet_col=0, facet_labels=["Second Digit", "Third Digit"],
    height=400,
    width=700, save_path="images/activation_patching/attn_2_Ai_1.pdf"
)

###################################
# 3 - CORRUPTING PREVIOUS LETTERS #
###################################

act_patch_resid_pre_iter = []
indices = [8, 9, 10]
for i, j in enumerate(indices):
    # Corrupt just the current word
    corrupted_tokens_i = clean_tokens.clone()
    corrupted_tokens_i[:, 8:j+1] = corrupted_tokens[:, 8:j+1] 
    _, corrupted_cache_i = model.run_with_cache(corrupted_tokens_i)
    compute_logit_diff_aux = partial(compute_logit_diff_2, answer_tokens=answer_tokens, average=False) # returns (batch_size, 3)
    compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits)[:, i].mean()
    act_patch_resid_pre = patching.get_act_patch_resid_pre(model, clean_tokens, corrupted_cache_i, compute_logit_diff_iter)
    act_patch_resid_pre_iter.append(act_patch_resid_pre)
act_patch_resid_pre_iter = torch.stack(act_patch_resid_pre_iter, dim=0)

baseline_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

imshow(
    act_patch_resid_pre_iter - baseline_logit_diff[..., None, None],
    x=labels,
    facet_col=0, # This argument tells plotly which dimension to split into separate plots
    facet_col_wrap=3,
    facet_labels=facet_labels, # Subtitles of separate plots
    title="Residual Stream Patching", 
    labels={"x": "Sequence Position", "y": "Layer"},
    height=400,
    width=1000, save_path="images/activation_patching/res_3.pdf"
)