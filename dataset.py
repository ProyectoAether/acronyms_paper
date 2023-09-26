from wonderwords import RandomWord
import string
import random
import torch

from transformer_lens import HookedTransformer

from utils import tokenize_single_letters


def generate_sentences(model, n=10):
    """
    Generate sentences of the form `The A B C (ABC) and the X Y Z (XY`.
    Returns
    -------
    `clean_final_tokens`: `torch.Tensor` of shape `(n, 18)` containing the tokenized sentences
    `clean_answer_tokens`: `torch.Tensor` of shape `(n, 3)` containing the tokenized acronym
    """ 
    acronym_letters = random.choices(string.ascii_uppercase, k=3)
    acronym_letters_space = [" " + c for c in acronym_letters]
    context_tokens = torch.cat([model.to_tokens("The" + ''.join(acronym_letters_space) + " ("),
                                tokenize_single_letters(model, acronym_letters),
                                model.to_tokens(") and the", prepend_bos=False)], dim=1)

    clean_final_tokens = []
    clean_answer_tokens = []

    for _ in range(n):
        acronym_letters = random.choices(string.ascii_uppercase, k=3)
        acronym_letters_space = [" " + c for c in acronym_letters]
        clean_final_tokens.append(model.to_tokens(acronym_letters_space + [" ("] + acronym_letters[:2], prepend_bos=False).squeeze())
        clean_answer_tokens.append(model.to_tokens(acronym_letters, prepend_bos=False).squeeze())

    clean_final_tokens = torch.stack(clean_final_tokens, dim=0)
    clean_final_tokens = torch.cat([context_tokens.expand(clean_final_tokens.shape[0], -1), clean_final_tokens], dim=-1)
    clean_answer_tokens = torch.stack(clean_answer_tokens, dim=0)

    # The word_idx is not really needed, as the position of each relevant token remains the same across all sentences, but we do it for 
    # compatibility reasons with the rest of the funcions, that word with more general datasets
    word_idx = {}
    word_idx["A1"] = torch.tensor([12] * n, device=clean_final_tokens.device)
    word_idx["A2"] = torch.tensor([13] * n, device=clean_final_tokens.device)
    word_idx["A3"] = torch.tensor([14] * n, device=clean_final_tokens.device)
    word_idx["END"] = torch.tensor([15] * n, device=clean_final_tokens.device)

    return clean_final_tokens, clean_answer_tokens, word_idx


def build_word_idx(model, tokens, answer_tokens_space):
    """
    Build a dictionary `word_idx`, which contains the following:
        - `word_idx["A1"]`: Tensor[batch_size]
            Contains the position of the first capital letter of the acronym. 
            We have the same for `"A2"` and `"A3"`
        - `word_idx["END"]`: Tensor[batch_size]
            Contains the position of the `(` token for every sentence
    Parameters
    -----------
    - `tokens`: Tensor[batch, max_len]
    - `answer_tokens_space`: Tensor[batch, 3]
        Contains the tokens of the letters of the acronym PRECEDED BY A SPACE (i.e. | X|)
    """

    word_idx = dict()

    batch_size = tokens.shape[0]

    idx_A1 = torch.zeros((batch_size,), dtype=torch.long, device=tokens.device)
    x, y = (tokens == answer_tokens_space[:, 0, None]).nonzero(as_tuple=True)
    idx_A1[x] = y

    idx_A2 = torch.zeros((batch_size,), dtype=torch.long, device=tokens.device)
    x, y = (tokens == answer_tokens_space[:, 1, None]).nonzero(as_tuple=True)
    idx_A2[x] = y

    idx_A3 = torch.zeros((batch_size,), dtype=torch.long, device=tokens.device)
    x, y = (tokens == answer_tokens_space[:, 2, None]).nonzero(as_tuple=True)
    idx_A3[x] = y

    idx_END = torch.zeros((batch_size,), dtype=torch.long, device=tokens.device)
    parens_token = model.to_tokens(" (", prepend_bos=False)
    x, y = (tokens == parens_token).nonzero(as_tuple=True)
    idx_END[x] = y

    word_idx["A1"] = idx_A1; word_idx["A2"] = idx_A2; word_idx["A3"] = idx_A3; word_idx["END"] = idx_END

    return word_idx


def get_random_acronym(model: HookedTransformer, max_iter=50):
    """
    Returns a random acronym sentence and its respective acronym (e.g. `"The Vivo Pregnancy Nerve ("`, `"VPN"`) by:
        Selecting three random words, with the constraint
        that the first capital letter of each word is represented as a single token together with its preceding space
        (e.g. `"|The| V|ivo| P|regnancy| N|erve|"`)
    """
    r = RandomWord()

    characters = string.ascii_lowercase

    acronym = ""
    example_prompt = []

    for _ in range(3):
        # Obtain a random character
        c = random.choice(characters)
        # Obtain a random word that starts with the acronym letter. 
        word = r.word(starts_with=c).capitalize()
        # Prepend a space
        word = " " + word
        first_token = model.to_str_tokens(word, prepend_bos=False)[0]
        # Keep iterating until we obtain a word that has its first letter tokenized as a single token
        iter = 0
        while first_token != word[:2] and iter < max_iter:
            word = r.word(starts_with=c).capitalize()
            word = " " + word
            first_token = model.to_str_tokens(word, prepend_bos=False)[0]
            iter += 1
        if iter == max_iter: continue
        acronym += c.upper()
        example_prompt.append(word)
    example_prompt = "The" + "".join(example_prompt) + " ("
    return example_prompt, acronym


def predict_acronym(model: HookedTransformer, prompt):
    predicted_acronym = ""
    tokens = model.to_tokens(prompt) # (batch, seq_len)
    for i in range(3):
        logits = model(tokens)
        predicted_token = logits[:, -1].max(dim=-1).indices # (batch)
        predicted_acronym += model.to_string(predicted_token)
        if i < 2:
            tokens = torch.cat([tokens, predicted_token[None]], dim=-1)
    return predicted_acronym


def find_proper_example(model: HookedTransformer):
    """
    Finds an acronym example that meets the following characteristics:
    1. The capital letters are tokenized only with its preceded space (e.g. | X|).
    This is acomplished via the `get_random_acronym` function.
    2. The acronym is predicted in exactly three steps by the model (e.g. |W|, |K|, |U|)
    3. The acronym is correctly predicted by the model
    """
    prompt, acronym = get_random_acronym(model)
    # Perform the prediction of the three next tokens
    pred_acronym = predict_acronym(model, prompt)
    # Keep doing this until we obtain a correct prediction
    while acronym != pred_acronym:
        prompt, acronym = get_random_acronym(model)
        pred_acronym = predict_acronym(model, prompt)
        #print(f"Prompt: {prompt}\nGT: {acronym}\tPred: {pred_acronym}\n")
    return prompt, acronym


def get_corrupted_tokens(model, clean_tokens, clean_answers, word_idx, random_acronym):
    """
    Given `tokens`, it corrupts it by randomly replacing the capital letters.
    Parameters
    ----------
    - `clean_tokens`: Tensor[batch_size, max_len]
    - `clean_answers`: List[string]

    Returns
    --------
    - `corrupted_tokens`: Tensor[batch_size, max_len]
    - `corrupted_answer_tokens`: Tensor[batch_size, 3]
    - `corrupted_answer_tokens_space`: Tensor[batch_size, 3]
    """

    corrupted_tokens = torch.zeros_like(clean_tokens)
    corrupted_answer_tokens_space = torch.zeros((len(clean_answers), 3), dtype=torch.long, device=corrupted_tokens.device)

    for i in range(clean_tokens.shape[0]):
        # List of possible tokens (exclude the three from the acronyms)
        possible_tokens = torch.cat([model.to_tokens(f" {c}", prepend_bos=False) for c in string.ascii_uppercase if c not in clean_answers[i]], dim=0).squeeze()
        acronym_replacement = possible_tokens[torch.randint(0, possible_tokens.shape[0], (3,))]
        corrupted_token = clean_tokens[None, i, :].clone()
        corrupted_token[:, word_idx["A1"][i]] = acronym_replacement[0]
        corrupted_token[:, word_idx["A2"][i]] = acronym_replacement[1]
        corrupted_token[:, word_idx["A3"][i]] = acronym_replacement[2]
        # If we pass to string and reencode, we want it to be the same, i.e. its capital letters will be naturally encoded as | X|
        retransformed_token = model.to_tokens(model.to_string(corrupted_token), prepend_bos=False)
        while (retransformed_token.shape != corrupted_token.shape) or (retransformed_token != corrupted_token).any():
            acronym_replacement = possible_tokens[torch.randint(0, possible_tokens.shape[0], (3,))]
            corrupted_token = clean_tokens[None, i, :].clone()
            corrupted_token[:, word_idx["A1"][i]] = acronym_replacement[0]
            corrupted_token[:, word_idx["A2"][i]] = acronym_replacement[1]
            corrupted_token[:, word_idx["A3"][i]] = acronym_replacement[2]
            retransformed_token = model.to_tokens(model.to_string(corrupted_token), prepend_bos=False)
        corrupted_tokens[i] = corrupted_token[0]
        if not random_acronym:
            corrupted_answer_tokens_space[i] = acronym_replacement[None, :]
        else:
            corrupted_answer_tokens_space[i] = possible_tokens[torch.randint(0, possible_tokens.shape[0], (3,))][None, :]
    corrupted_answer_tokens = torch.cat([tokenize_single_letters(model, s) for s in [s.replace(" ", "") for s in model.to_string(corrupted_answer_tokens_space)]], dim=0)
    
    
    return corrupted_tokens, corrupted_answer_tokens, corrupted_answer_tokens_space


class AcronymsDataset:
    """
    Dataset class that contains examples of acronyms that meet the following:
        1. The initial letter of every word of the acronym is tokenized only with its preceding space, e.g. `The| W|ound| K|ettle...`. 
        2. The prediction made by the model (GPT-2 Small) is correct
        3. The prediction should be performed in exactly 3 steps, i.e. letter by letter.
    For every "clean" example, we also provide a "corrupted" counterpart, formed by replacing the capital letters while keeping them equally tokenized
    (i.e. `| X|`).
    Parameters:
    -----------
    - `random_corrupted_acronyms`: `bool`
        If `True`, the corrupted examples will have a random appended acronym, i.e. 
        "The Eound Wettle Fptight (XY" instead of "The Eound Wettle Fptight (EW"

    Attributes:
    -----------
    - `clean_prompts`: List[str]
        List of the prompts in string format, e.g. `"The Wound Kettle Uptight ("`
    - `clean_answers`: List[str]
        List of the acronyms in string format, e.g. `"WKU"`
    - `clean_tokens`: Tensor[batch, max_len]
        Tokenized prompts
    - `clean_answer_token`: Tensor[batch, 3]
        Tokenized acronyms (a token for every letter)
    
    The `corrupted` counterparts have the same format.
    """
    def __init__(self, model, acronyms_file="acronyms.txt", n_context=0, context_file="acronyms_context.txt", random_corrupted_acronyms=True):
        # LOAD THE ACRONYMS
        with open(acronyms_file, "r") as f:
            clean_sentences = f.read().splitlines()
        clean_prompts, clean_answers = zip(*[s.split("(") for s in clean_sentences])
        clean_prompts = [("t" if n_context > 0 else "T") + s[1:] + "(" for s in clean_prompts]
        # LOAD THE CONTEXT
        if n_context > 0:
            with open(context_file, "r") as f:
                context = f.read().splitlines()
                context = [("t" if i > 0 else "T") + s[1:] + ")" for i, s in enumerate(context)]
            context_prompt = ", ".join(context[:n_context]) + " and "
            context_tokens = model.to_tokens(context_prompt)


        # Add two extra tokens at the end for the next two letters of the acronym
        clean_tokens = model.to_tokens(clean_prompts) # (batch, max_len)
        clean_tokens = torch.cat([clean_tokens, 50256 * torch.ones((clean_tokens.shape[0], 2), dtype=torch.long, device=clean_tokens.device)], dim=-1) # (batch, max_len + 2)
        clean_answer_tokens = torch.cat([tokenize_single_letters(model, clean_answer) for clean_answer in clean_answers], dim=0)
        clean_answer_tokens_space = torch.cat([tokenize_single_letters(model, clean_answer, prepend_space=True) for clean_answer in clean_answers], dim=0)
        word_idx = build_word_idx(model, clean_tokens, clean_answer_tokens_space)
        corrupted_tokens, corrupted_answer_tokens, corrupted_answer_tokens_space = get_corrupted_tokens(model, clean_tokens, clean_answers, word_idx, random_acronym=random_corrupted_acronyms)

        # Append the context (if any)
        if n_context > 0:
            clean_prompts = [context_prompt + s for s in clean_prompts]

            clean_tokens = torch.cat([context_tokens.expand(clean_tokens.shape[0], -1), clean_tokens[:, 1:]], dim=-1)
            corrupted_tokens = torch.cat([context_tokens.expand(corrupted_tokens.shape[0], -1), corrupted_tokens[:, 1:]], dim=-1)

            # Update word_idx
            n_context_tokens = context_tokens.shape[1] - 1
            word_idx["A1"] += n_context_tokens
            word_idx["A2"] += n_context_tokens
            word_idx["A3"] += n_context_tokens
            word_idx["END"] += n_context_tokens

        # Save the tokens for the prompts of the last iteration, i.e. "The Wound Kettle Uptight (WK"
        clean_final_tokens = clean_tokens.clone()
        clean_final_tokens[torch.arange(clean_tokens.shape[0]), word_idx["END"] + 1] = clean_answer_tokens[:, 0]
        clean_final_tokens[torch.arange(clean_tokens.shape[0]), word_idx["END"] + 2] = clean_answer_tokens[:, 1]
        corrupted_final_tokens = corrupted_tokens.clone()
        corrupted_final_tokens[torch.arange(corrupted_tokens.shape[0]), word_idx["END"] + 1] = corrupted_answer_tokens[:, 0]
        corrupted_final_tokens[torch.arange(corrupted_tokens.shape[0]), word_idx["END"] + 2] = corrupted_answer_tokens[:, 1]

        self.clean_prompts = clean_prompts
        self.clean_tokens = clean_tokens
        self.clean_final_tokens = clean_final_tokens
        self.clean_answer_tokens = clean_answer_tokens
        self.corrupted_prompts = [context_prompt + s for s in model.to_string(corrupted_tokens)] if n_context > 0  else model.to_string(corrupted_tokens)
        self.corrupted_tokens = corrupted_tokens
        self.corrupted_final_tokens = corrupted_final_tokens
        self.corrupted_answer_tokens = corrupted_answer_tokens
        self.word_idx = word_idx
        self.n_samples = len(clean_prompts)


def randomly_replace_capitals(model, clean_final_tokens, word_idx, replace_acronym=False):
    """
    Corrupts the sentences by applying method #2, i.e. randomly replacing the capital letters.
    For example: `"The A B C (AB" -> "The X Y Z (AB"`

    Parameters
    ----------
    - `clean_final_tokens`: Tensor[batch_size, seq_len]
    - `word_idx`: `dict`
    """
    batch_size = clean_final_tokens.shape[0]
    corrupted_final_tokens = clean_final_tokens.clone()

    for a in ["A1", "A2", "A3"]:
        corrupted_final_tokens[torch.arange(batch_size), word_idx[a]] = \
            model.to_tokens([" " + c for c in random.choices(string.ascii_uppercase, k=batch_size)], prepend_bos=False).squeeze()
    
    # if indicated, replace also the letters of the acronym
    if replace_acronym:
        corrupted_final_tokens[torch.arange(batch_size), word_idx["END"]+1] = \
            model.to_tokens(random.choices(string.ascii_uppercase, k=batch_size), prepend_bos=False).squeeze()
        corrupted_final_tokens[torch.arange(batch_size), word_idx["END"]+2] = \
            model.to_tokens(random.choices(string.ascii_uppercase, k=batch_size), prepend_bos=False).squeeze()
    
    return corrupted_final_tokens