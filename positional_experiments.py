import argparse
import time
from functools import partial
from itertools import combinations

import einops
import plotly.express as px
import torch
from IPython.display import clear_output
# TransformerLens
from transformer_lens import HookedTransformer, patching

from dataset import *
from mechanistic_utils import *
from plotly_utils import imshow, scatter
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

def swap_pos_embed(activations, hook, a=2, b=6):
    # activations has shape (batch_size, seq_len, d_model)
    pos_embed_C1 = activations[:, a].clone()
    pos_embed_C2 = activations[:, b].clone()
    activations[:, b] = pos_embed_C1
    activations[:, a] = pos_embed_C2
    return activations

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


letter_mover_heads = [[8, 11], [9, 9], [10, 10], [11, 4]]
idx = [0, 2, 4, 6]

for layer, head in letter_mover_heads:
    for a, b in combinations([1, 2, 3], 2):
        

        attn_df = get_attn_df(layer, head, clean_cache)

        hook_fn = partial(swap_pos_embed, a=idx[a], b=idx[b])
        model.add_hook("hook_pos_embed", hook_fn)
        pos_corrupted_logits, pos_corrupted_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks(including_permanent=True)

        pos_corrupted_attn_df = get_attn_df(layer, head, pos_corrupted_cache)

        attn_df["Experiment"] = "Clean Run"
        pos_corrupted_attn_df["Experiment"] = "Swap"

        attn_df = pd.concat([attn_df, pos_corrupted_attn_df])

        attn_df = attn_df[(attn_df["Token"] == "C1") | (attn_df["Token"] == "C2") | (attn_df["Token"] == "C3")]

        fig = px.bar(attn_df, x="Token", y = "Attention Probability", error_y="std",
            barmode="group", color="Experiment", width=800, height=400, facet_col="Letter",
            title=f"Attention probabilities for head {layer}.{head} swapping positions C{a} <-> C{b}")

        fig.write_image(f"images/positional_experiments/swap_pos/swap_pos_C{a}_C{b}_{layer}_{head}.pdf")
        time.sleep(1)
        fig.write_image(f"images/positional_experiments/swap_pos/swap_pos_C{a}_C{b}_{layer}_{head}.pdf")

#################################################
# POSITIONAL (SWAPPING BOS) ACTIVATION PATCHING #
#################################################

def replace_head_activations(activations, hook, head_idx, new_cache):
    # Replace head activations with the ones stored in `new_cache` (Only the attention to BOS token!)
    activations[:, head_idx, :, 0] = new_cache[hook.name][:, head_idx, :, 0]
    return activations


def run_with_cache_and_swap(model, tokens, a=2, b=6):
    pattern_name_filter = lambda name: name.endswith("pattern")
    _, swapped_cache = model.run_with_cache(tokens, names_filter=pattern_name_filter, return_type=None)

    for layer in range(model.cfg.n_layers):
        Ca = swapped_cache["pattern", layer][:, :, a, 0].clone()
        Cb = swapped_cache["pattern", layer][:, :, b, 0].clone()
        swapped_cache["pattern", layer][:, :, b, 0] = Ca
        swapped_cache["pattern", layer][:, :, a, 0] = Cb

    return swapped_cache


def get_act_patch_attn_head_BOS_cache(model: HookedTransformer, orig_tokens, metric, a, b):
    act_patch_attn_head_BOS = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=orig_tokens.device)
    swapped_cache = run_with_cache_and_swap(model, orig_tokens, a=a, b=b)
    for layer, head in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):
        hook_fn = partial(replace_head_activations, head_idx=head, new_cache=swapped_cache)
        model.add_hook(utils.get_act_name("pattern", layer), hook_fn)
        swapped_logits = model(orig_tokens)
        model.reset_hooks(including_permanent=True)
        act_patch_attn_head_BOS[layer, head] = metric(swapped_logits)
    
    return act_patch_attn_head_BOS


for a, b in combinations([1, 2, 3], 2):
    act_patch_attn_head_BOS_iter = []

    for i in range(3):
        compute_logit_diff_aux = partial(compute_logit_diff_2, answer_tokens=answer_tokens, average=False) # returns (batch_size, 3)
        compute_logit_diff_iter = lambda logits: compute_logit_diff_aux(logits)[:, i].mean()
        act_patch_attn_head_BOS = get_act_patch_attn_head_BOS_cache(model, clean_tokens, compute_logit_diff_iter, a=idx[a], b=idx[b])
        act_patch_attn_head_BOS_iter.append(act_patch_attn_head_BOS)
    act_patch_attn_head_BOS_iter = torch.stack(act_patch_attn_head_BOS_iter, dim=0)

    facet_labels = ["Letter 1", "Letter 2", "Letter 3"]

    baseline_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

    imshow(
        act_patch_attn_head_BOS_iter - baseline_logit_diff[..., None, None],
        facet_col=0,
        facet_labels=facet_labels,
        labels={"y": "Layer", "x": "Head"}, 
        title=f"Swapping Attention to BOS C{a} <-> C{b}",
        width=800,
        height=400, save_path=f"images/positional_experiments/swap_BOS/swap_BOS_C{a}_C{b}_attn_patching.pdf"
    )

    ############################################################################
    # PATCHING IMPORTANT HEADS AND VISUALIZING CHANGE IN ATTN. PATTERN OF 8.11 #
    ############################################################################

    path_patch_head_to_heads_iteration = act_patch_attn_head_BOS_iter - baseline_logit_diff[..., None, None]
    threshold = 5e-3

    top_heads_1 = torch.argwhere(-1*path_patch_head_to_heads_iteration[0] > threshold).tolist()
    top_heads_2 = torch.argwhere(-1*path_patch_head_to_heads_iteration[1] > threshold).tolist()
    top_heads_3 = torch.argwhere(-1*path_patch_head_to_heads_iteration[2] > threshold).tolist()

    top_heads = list(set(tuple(head) for head in (top_heads_1 + top_heads_2 + top_heads_3)))
    heads_to_patch = top_heads


    for layer, head in letter_mover_heads:
        attn_df = get_attn_df(layer, head, clean_cache)
        swapped_cache = run_with_cache_and_swap(model, clean_tokens, a=idx[a], b=idx[b])

        # ONLY POSITIONAL SWAPPING
        hook_fn = partial(swap_pos_embed, a=idx[a], b=idx[b])
        model.add_hook("hook_pos_embed", hook_fn)
        _, pos_corrupted_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks(including_permanent=True)

        pos_corrupted_attn_df = get_attn_df(layer, head, pos_corrupted_cache)
        attn_df["Experiment"] = "Clean Run"
        pos_corrupted_attn_df["Experiment"] = "Swap POS"
        attn_df = pd.concat([attn_df, pos_corrupted_attn_df])

        # ONLY BOS SWAPPING
        for layer_i, head_i in heads_to_patch:
            hook_fn = partial(replace_head_activations, head_idx=head_i, new_cache=swapped_cache)
            model.add_hook(utils.get_act_name("pattern", layer_i), hook_fn)
        _, pos_corrupted_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks(including_permanent=True)

        pos_corrupted_attn_df = get_attn_df(layer, head, pos_corrupted_cache)
        pos_corrupted_attn_df["Experiment"] = "Swap BOS"
        attn_df = pd.concat([attn_df, pos_corrupted_attn_df])

        # POS + BOS SWAPPING
        hook_fn = partial(swap_pos_embed, a=idx[a], b=idx[b])
        model.add_hook("hook_pos_embed", hook_fn)
        for layer_i, head_i in heads_to_patch:
            hook_fn = partial(replace_head_activations, head_idx=head_i, new_cache=swapped_cache)
            model.add_hook(utils.get_act_name("pattern", layer_i), hook_fn)
        _, pos_corrupted_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks(including_permanent=True)

        pos_corrupted_attn_df = get_attn_df(layer, head, pos_corrupted_cache)
        pos_corrupted_attn_df["Experiment"] = "Swap POS+BOS"
        attn_df = pd.concat([attn_df, pos_corrupted_attn_df])

        attn_df = attn_df[(attn_df["Token"] == "C1") | (attn_df["Token"] == "C2") | (attn_df["Token"] == "C3")]

        fig = px.bar(attn_df, x="Token", y = "Attention Probability", error_y="std",
        barmode="group", color="Experiment", width=1000, height=400, facet_col="Letter",
        title=f"Attn. probs. for head {layer}.{head} when swapping POS/BOS tokens of words C{a} <-> C{b}")
        fig.write_image(f"images/positional_experiments/swap_BOS/BOS_{layer}_{head}_{a}_{b}.pdf")
        time.sleep(1)
        fig.write_image(f"images/positional_experiments/swap_BOS/BOS_{layer}_{head}_{a}_{b}.pdf")