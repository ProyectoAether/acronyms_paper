import pandas as pd
import numpy as np
import torch
from transformer_lens import utils
import string
from bokeh.plotting import figure

device = "cuda" if torch.cuda.is_available() else "cpu"

capital_letters_tokens = torch.tensor([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
         50, 51, 52, 53, 54, 55, 56, 57], dtype=torch.long, device=device)

def tokenize_single_letters(model, s, prepend_space=False):
    """
    Given a string, tokenizes its letters on single tokens
    For example, "ABC" -> "|A|B|C|" 
    If `prepend_space=True`, returns `"ABC"` -> `"| A| B| C|"` 
    """
    if prepend_space:
        return torch.cat([model.to_tokens(" " + s[i], prepend_bos=False) for i in range(len(s))], dim=-1)
    else:
        return torch.cat([model.to_tokens(s[i], prepend_bos=False) for i in range(len(s))], dim=-1)

def compute_logit_diff(logits, correct_answer_token, word_idx, iteration=0, average=True):
    """
    Given the logits and the token of the correct answer, computes the logit 
    difference between the correct answer and the mean logits over the rest of 
    possible capital letters

    logits (batch, seq_len, d_vocab)
    correct_answer_token (batch, 1)
    """
    batch_size = logits.shape[0]
    # Take logits of the final prediction
    logits = logits[torch.arange(batch_size), (word_idx["END"] + iteration)[:batch_size]] # (batch, d_vocab)
    capital_letters_expanded = capital_letters_tokens[None, :].expand(batch_size, -1)
    incorrect_answer_tokens = capital_letters_expanded[correct_answer_token != capital_letters_expanded].reshape(batch_size, -1)
    logit_diff = logits.gather(-1, correct_answer_token).squeeze() - logits.gather(-1, incorrect_answer_tokens).mean(-1)
    logit_diff = logit_diff.squeeze() # (batch, )
    return logit_diff.mean(0, keepdim=True)  if average else logit_diff # (batch,)


def compute_logit_diff_2(logits, answer_tokens, average=True):
    """
    Compute the logit difference between the correct answer and the largest logit
    of all the possible incorrect capital letters. This is done for every iteration
    (i.e. each of the three letters of the acronym) and then averaged if desired.
    If `average=False`, then a `Tensor[batch_size, 3]` is returned, containing the
    logit difference at every iteration for every prompt in the batch

    Parameters:
    -----------
    - `logits`: `Tensor[batch_size, seq_len, d_vocab]`
    - `answer_tokens`: Tensor[batch_size, 3]
    """
    # Logits of the correct answers (batch_size, 3)
    correct_logits = logits[:, -3:].gather(-1, answer_tokens[..., None]).squeeze()
    # Retrieve the maximum logit of the possible incorrect answers
    capital_letters_tokens = torch.tensor([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
         50, 51, 52, 53, 54, 55, 56, 57], dtype=torch.long, device=device)
    batch_size = logits.shape[0]
    capital_letters_tokens_expanded = capital_letters_tokens.expand(batch_size, 3, -1)
    incorrect_capital_letters = capital_letters_tokens_expanded[capital_letters_tokens_expanded != answer_tokens[..., None]].reshape(batch_size, 3, -1)
    incorrect_logits, _ = logits[:, -3:].gather(-1, incorrect_capital_letters).max(-1)
    # Return the mean
    return (correct_logits - incorrect_logits).mean() if average else (correct_logits - incorrect_logits)


def compute_score(logits,
                  correct_answer_token, 
                  clean_logits,
                  corrupted_logits,
                  word_idx,
                  iteration=0,
                  denoising=True):
    """
    If `denoising=True`, returns a value between 0 and 1, 0 meaning that
    the logit diff is equal to the corrupted run, and 1 meaning that it has 
    completely recovered the performance of the clean run.
    If `denoising=False`, then it returns a value between 0 and -1, 0 meaning
    that the performance has not degraded, and -1 that the performance has completely
    degraded to the level of the corrupted run.
    - `logits`: Tensor[batch_size, seq_len, d_vocab]
    - `correct_answer_token`: Tensor[batch_size, 1]
    - `clean_logits` : Tensor[batch_size, seq_len, d_vocab]
    - `corrupted_logits`: Tensor[batch_size, seq_len, d_vocab]
    - `word_idx`: see `AcronymDataset`
    - `iteration`: int (0-2)
        current iteration
    """
    score = compute_logit_diff(logits, correct_answer_token, word_idx, iteration=iteration, average=True) # (1,)
    clean_score = compute_logit_diff(clean_logits, correct_answer_token, word_idx, iteration=iteration, average=True) # (1,)
    corrupted_score = compute_logit_diff(corrupted_logits, correct_answer_token, word_idx, iteration=iteration, average=True) # (1,)
    if denoising:
        return (score - corrupted_score) / (clean_score - corrupted_score)
    else:
        return -1*(score - clean_score) / (corrupted_score - clean_score)


def compute_avg_logit_diff(logits, correct_answer_tokens, word_idx, average=True):
    """
    Auxiliary function that just computes `compute_logit_diff` for every iteration and
    returns the average value

    Parameters
    ----------
    - logits (batch, seq_len, d_vocab)
    - correct_answer_token (batch, 3)
    """
    logit_diff = 0.
    for i in range(3):
        logit_diff += compute_logit_diff(logits, correct_answer_tokens[:, i][:, None], word_idx, iteration=i, average=average)
    
    return logit_diff / 3.

def topk_of_Nd_tensor(tensor, k):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

def plot_logit_distribution_capital(example_prompt, end_pos, model):
    """
    Given a prompt of the type `The Wound Kettle Uptight (` plots the logit
    distribution over the capital letters
    """
    capital_letters = list(string.ascii_uppercase)
    capital_letters_tokens = model.to_tokens(capital_letters, prepend_bos=False).squeeze()

    logits = model(example_prompt)[0, end_pos]
    capital_letters_logits = logits[capital_letters_tokens]

    p = figure(x_range=capital_letters, height=350, title=f"Logits of Capital Letters for prompt: {example_prompt}",
            x_axis_label="Tokens", y_axis_label="Logits")

    p.vbar(x=capital_letters, top=capital_letters_logits.cpu().numpy(), width=0.9)
    p.line([-1e2, 1e2], [logits.mean().item(), logits.mean().item()], legend_label="Avg. Logit", line_color="red", line_width=3)
    # display legend in top left corner (default is top right corner)
    p.legend.location = "bottom_right"

    return p


def attn_every_letter_iteration(cache, important_heads, word_idx):
    """
    Given a cache, returns a DataFrame with the attention paid to every capital
    letter on every iteration 
    """
    attn_patterns = torch.stack([
            cache["pattern", layer][:, head]
            for layer, head in important_heads
        ])

    attn_df = pd.DataFrame()

    i = 0
    for i in range(3):
        # Average attention paid to A1, A2 and A3 on iteration i
        attn_a1 = attn_patterns[:, torch.arange(attn_patterns.shape[1]), word_idx["END"] + i, word_idx["A1"]].mean(-1)
        attn_a1_std = attn_patterns[:, torch.arange(attn_patterns.shape[1]), word_idx["END"] + i, word_idx["A1"]].std(-1)

        attn_a2 = attn_patterns[:, torch.arange(attn_patterns.shape[1]), word_idx["END"] + i, word_idx["A2"]].mean(-1)
        attn_a2_std = attn_patterns[:, torch.arange(attn_patterns.shape[1]), word_idx["END"] + i, word_idx["A2"]].std(-1)

        attn_a3 = attn_patterns[:, torch.arange(attn_patterns.shape[1]), word_idx["END"] + i, word_idx["A3"]].mean(-1)
        attn_a3_std = attn_patterns[:, torch.arange(attn_patterns.shape[1]), word_idx["END"] + i, word_idx["A3"]].std(-1)

        attn_as = torch.cat([attn_a1, attn_a2, attn_a3], dim=0).cpu().numpy()
        attn_as_std = torch.cat([attn_a1_std, attn_a2_std, attn_a3_std], dim=0).cpu().numpy()

        attn_df_i = pd.DataFrame({"Head": ["8.11", "9.9", "10.10"]*3,
                                "Token":  [f"A1"]*3 + [f"A2"]*3 + [f"A3"]*3,
                                "Attention Probability": attn_as,
                                "std": attn_as_std,
                                "Iteration": i})
        attn_df = pd.concat([attn_df, attn_df_i])
    return attn_df