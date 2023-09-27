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

corrupted_tokens = clean_tokens.clone()
corrupted_tokens = corrupted_tokens[torch.randperm(corrupted_tokens.shape[0])]

corrupted_tokens_acronym = clean_tokens.clone()
corrupted_tokens_acronym = corrupted_tokens_acronym[torch.randperm(corrupted_tokens_acronym.shape[0])]

corrupted_tokens[:, -2:] = corrupted_tokens_acronym[:, -2:]

corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

capital_letters = list(string.ascii_uppercase)
capital_letters_space = [" " + c for c in string.ascii_uppercase]

capital_letters_tokens = model.to_tokens(capital_letters, prepend_bos=False)[:, 0]
capital_letters_space_tokens = model.to_tokens(capital_letters_space, prepend_bos=False)[:, 0]

def contribution_OV_circuit(heads, input_vocab, output_vocab):
    full_OV_circuit = torch.zeros((input_vocab.shape[0], output_vocab.shape[0]), device=input_vocab.device)
    for layer, head in heads:
        full_OV_circuit += ((model.embed.W_E[input_vocab] @ model.OV[layer, head] @ model.unembed.W_U)[:, output_vocab]).AB
    return full_OV_circuit

#heads = [[8, 11], [9, 9], [10, 10], [11, 4]]
heads = [[8, 11]]
full_OV_circuit = []
for input_vocab, output_vocab in itertools.product([capital_letters_tokens, capital_letters_space_tokens], [capital_letters_tokens, capital_letters_space_tokens]):
    full_OV_circuit.append(contribution_OV_circuit(heads, input_vocab, output_vocab))

full_OV_circuit = torch.stack(full_OV_circuit, dim=0)
facet_labels = ["|X| -> |X|", "|X| -> |_X|", "|_X| -> |X|", "|_X| -> |_X|"]

imshow(full_OV_circuit, x=capital_letters, y=capital_letters, 
       facet_col=0, facet_labels=facet_labels, facet_col_wrap=4, labels={"x": "Output", "y": "Input"},
       title=f"Full OV circuit for head{'s' if len(heads) > 1 else ''} {heads}", width=800, height=350, save_path=f"images/letter_mover_heads/OV_{len(heads)}.pdf")


head_to_plot = [[8, 11], [9, 9], [10, 10], [11, 4]]

for layer, head in head_to_plot:
    print(f"Plotting head {layer}.{head}...")
    # Retrieve the attention patterns A(i-1)->C
    attn_probs = clean_cache["pattern", layer][:, head, -3:, [2, 4, 6]] # (batch_size, q_pos, k_pos)
    attn_probs = einops.rearrange(attn_probs,
        "batch_size iter token -> iter token batch_size")
    # Map the residual stream vector into the logits of the proper tokens
    z = clean_cache[utils.get_act_name("z", layer)][:, :, head]
    output = (z @ model.W_O[layer, head])[:, -3:]
    # Unembed
    output = (output @ model.W_U)
    # Gather the proper logits
    #gather_idx = clean_tokens[:, [2, 5, 8]][:, None, :].expand(-1, 3, -1)
    gather_idx = answer_tokens[:, None, :].expand(-1, 3, -1)
    logit_lens = output.gather(-1, gather_idx) # (batch_size, iter, token)
    logit_lens = einops.rearrange(logit_lens,
        "batch_size iter token -> iter token batch_size")

    df_attn_probs = pd.DataFrame()
    df_attn_probs["Attn. prob. on token"] = attn_probs.reshape(-1).cpu()
    df_attn_probs["Logits"] = logit_lens.reshape(-1).cpu()
    df_attn_probs["Letter"] = [1] * 3*n + [2] * 3*n + [3] * 3*n
    df_attn_probs["Ci"] = (["C1"] * n + ["C2"] * n + ["C3"]* n)*3

    fig = px.scatter(df_attn_probs, x="Attn. prob. on token", y="Logits",
           facet_col="Letter", color="Ci", symbol="Ci", width=1000, height=400,
           title=f"Projection of head {layer}.{head} onto the letter logits vs. attention probability")
    fig.update_traces(marker=dict(size=5))
    fig.write_image(f"images/letter_mover_heads/scatter_{layer}_{head}.pdf")
    time.sleep(1)
    fig.write_image(f"images/letter_mover_heads/scatter_{layer}_{head}.pdf")