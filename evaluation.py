import torch
import argparse
import time

# TransformerLens
from transformer_lens import HookedTransformer

import plotly.express as px

# My libraries
from utils import *
from mechanistic_utils import *
from dataset import *

torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=50, help="number of samples to use")
args = parser.parse_args()

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

n = args.n
prompts = prompts[:n]
acronyms = acronyms[:n]

clean_tokens = model.to_tokens(prompts)
answer_tokens = model.to_tokens(acronyms, prepend_bos=False)
clean_logits = model(clean_tokens)
clean_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens)
print(f"Average Logit Diff. (n={n}):\n\t{clean_logit_diff.item():.2f}")

corrupted_tokens = clean_tokens.clone()
corrupted_tokens = corrupted_tokens[torch.randperm(corrupted_tokens.shape[0])]
corrupted_tokens_acronym = clean_tokens.clone()
corrupted_tokens_acronym = corrupted_tokens_acronym[torch.randperm(corrupted_tokens_acronym.shape[0])]
corrupted_tokens[:, -2:] = corrupted_tokens_acronym[:, -2:]
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

def mean_ablate_head(activations, hook, head_idx, new_cache):
    # activation has shape (batch, pos, head, d_head)
    activations[:, :, head_idx] = new_cache[hook.name][:, :, head_idx].mean(0)[None, ...]
    return activations


circuit_heads = [[], [8, 11], [9, 9], [10, 10], [11, 4], [5, 8], [4, 11], [2, 2], [1, 0]]

circuit_heads_i = []
logit_diffs = []
std_logit_diffs = []

for circuit_head in circuit_heads:
    circuit_heads_i.append(circuit_head)
    heads_to_patch = [[a, b] for a, b in itertools.product(range(0, model.cfg.n_layers), range(model.cfg.n_heads)) if [a, b] not in circuit_heads_i]

    model.reset_hooks(including_permanent=True)
    for layer_i, head_i in heads_to_patch:
        hook_fn = partial(mean_ablate_head, head_idx=head_i, new_cache=corrupted_cache)
        model.add_hook(utils.get_act_name("z", layer_i), hook_fn)
    circuit_logits = model(clean_tokens)
    model.reset_hooks(including_permanent=True)

    logit_diff = compute_logit_diff_2(circuit_logits, answer_tokens, average=False)
    av_logit_diff = logit_diff.mean(0)
    std_logit_diff = logit_diff.std(0)
    logit_diffs.append(av_logit_diff)
    std_logit_diffs.append(std_logit_diff)
logit_diffs = torch.cat(logit_diffs, dim=0)
std_logit_diffs = torch.cat(std_logit_diffs, dim=0)

df_logit_diffs = pd.DataFrame()
df_logit_diffs["Logit Diff."] = logit_diffs.detach().cpu()
df_logit_diffs["Error"] = std_logit_diffs.detach().cpu()
df_logit_diffs["Component"] = 3*["None"] + [x for s in [3*[f"{layer}.{head}"] for layer, head in circuit_heads[1:]] for x in s]
df_logit_diffs["Letter"] = [1, 2, 3] * len(circuit_heads)

clean_logit_diff = compute_logit_diff_2(clean_logits, answer_tokens, average=False).mean(0)

fig = px.line(df_logit_diffs, x="Component", y="Logit Diff.", error_y="Error",
              facet_col="Letter", title="Logit Diff. vs. Progressively Adding Components", width=900, height=400)
fig.add_hline(y=clean_logit_diff[0].item(), line_width=1.5, line_dash="dash", line_color="black", row=1, col=1)
fig.add_hline(y=clean_logit_diff[1].item(), line_width=1.5, line_dash="dash", line_color="black", row=1, col=2)
fig.add_hline(y=clean_logit_diff[2].item(), line_width=1.5, line_dash="dash", line_color="black", row=1, col=3)
fig.write_image(f"images/evaluation/evaluation.pdf")
time.sleep(1)
fig.write_image(f"images/evaluation/evaluation.pdf")

