from functools import partial
import itertools
from tqdm import tqdm

import circuitsvis as cv
from IPython.display import display, HTML

import torch
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens import patching, utils
from transformer_lens.hook_points import HookPoint

from dataset import AcronymsDataset


def get_act_patch_block_every_iteration(model: HookedTransformer, 
                                        clean_tokens, corrupted_tokens, 
                                        clean_answer_tokens, corrupted_answer_tokens,
                                        word_idx, 
                                        metric):
    """
    Modification of the function `get_act_patch_block_every` to output the results
    of the three iterations of the acronym predicting task.
    
    - `clean_tokens` and `corrupted tokens`: Tensor[batch, seq_len]
    - `clean_answer_tokens` and `corrupted_answer_tokens`: Tensor[batch, 3]
    Returns a tensor of shape [n_iterations, 3, n_layers, seq_len] 
    """
    batch_size = clean_tokens.shape[0]
    seq_len = clean_tokens.shape[-1]
    # Clone the clean and corrupted tensors to avoid modifying the original ones
    clean_tokens = clean_tokens.clone()
    corrupted_tokens = corrupted_tokens.clone()
    act_patch_block_every_iteration = torch.zeros((3, 3, model.cfg.n_layers, seq_len))
    for i in range(clean_answer_tokens.shape[-1]):

        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

        compute_score_aux = partial(metric, 
                                correct_answer_token=clean_answer_tokens[:, i][:, None],
                                clean_logits=clean_logits, 
                                corrupted_logits=corrupted_logits,
                                word_idx=word_idx,
                                iteration=i,
                                denoising=True)
        act_patch_block_every_iteration[i] = \
            patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, compute_score_aux)

        # Unless we are at the last iteration, update the prompts
        if i < clean_answer_tokens.shape[-1] - 1:
            clean_tokens[torch.arange(batch_size), word_idx["END"][:batch_size] + i + 1] = clean_answer_tokens[:, i]
            corrupted_tokens[torch.arange(batch_size), word_idx["END"][:batch_size] + i + 1] = corrupted_answer_tokens[:, i]
            # clean_tokens = torch.cat([clean_tokens, clean_answer_tokens[:, i][:, None]], dim=-1)
            # corrupted_tokens = torch.cat([corrupted_tokens, corrupted_answer_tokens[:, i][:, None]], dim=-1)
    
    return act_patch_block_every_iteration


def get_act_patch_attn_head_out_all_pos_iteration(model: HookedTransformer, 
                                                  clean_tokens, corrupted_tokens, 
                                                  clean_answer_tokens, corrupted_answer_tokens,
                                                  word_idx,
                                                  metric):
    """
    Modification of the function `get_act_patch_attn_head_out_all_pos` to output the results
    of the three iterations of the acronym predicting task.
    
    - `clean_tokens` and `corrupted tokens`: Tensor[batch, seq_len]
    - `clean_answer_tokens` and `corrupted_answer_tokens`: Tensor[batch, 3]
    Returns a tensor of shape [n_iterations, n_layers, n_heads] 
    """
    batch_size = clean_tokens.shape[0]
    # Clone the clean and corrupted tensors to avoid modifying the original ones
    clean_tokens = clean_tokens.clone()
    corrupted_tokens = corrupted_tokens.clone()
    act_patch_attn_head_out_all_pos_iteration = torch.zeros((3, model.cfg.n_layers, model.cfg.n_heads))
    for i in range(clean_answer_tokens.shape[-1]):

        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

        compute_score_aux = partial(metric, 
                                correct_answer_token=clean_answer_tokens[:, i][:, None],
                                clean_logits=clean_logits, 
                                corrupted_logits=corrupted_logits,
                                word_idx=word_idx,
                                iteration=i,
                                denoising=True)
        act_patch_attn_head_out_all_pos_iteration[i] = \
            patching.get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, compute_score_aux)

        # Unless we are at the last iteration, update the prompts
        if i < clean_answer_tokens.shape[-1] - 1:
            clean_tokens[torch.arange(batch_size), word_idx["END"][:batch_size] + i + 1] = clean_answer_tokens[:, i]
            corrupted_tokens[torch.arange(batch_size), word_idx["END"][:batch_size] + i + 1] = corrupted_answer_tokens[:, i]
            # clean_tokens = torch.cat([clean_tokens, clean_answer_tokens[:, i][:, None]], dim=-1)
            # corrupted_tokens = torch.cat([corrupted_tokens, corrupted_answer_tokens[:, i][:, None]], dim=-1)
    
    return act_patch_attn_head_out_all_pos_iteration


def get_act_patch_attn_head_all_pos_every_iteration(model: HookedTransformer, 
                                        clean_tokens, corrupted_tokens, 
                                        clean_answer_tokens, corrupted_answer_tokens,
                                        word_idx, 
                                        metric):
    """
    Modification of the function `get_act_patch_attn_head_all_pos_every` to output the results
    of the three iterations of the acronym predicting task.
    
    - `clean_tokens` and `corrupted tokens`: Tensor[batch, seq_len]
    - `clean_answer_tokens` and `corrupted_answer_tokens`: Tensor[batch, 3]
    Returns a tensor of shape [n_iterations, 3, n_layers, seq_len] 
    """
    batch_size = clean_tokens.shape[0]
    # Clone the clean and corrupted tensors to avoid modifying the original ones
    clean_tokens = clean_tokens.clone()
    corrupted_tokens = corrupted_tokens.clone()

    act_patch_attn_head_all_pos_every_iteration = torch.zeros((3, 5, model.cfg.n_layers, model.cfg.n_heads))
    for i in range(clean_answer_tokens.shape[-1]):

        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

        compute_score_aux = partial(metric, 
                                correct_answer_token=clean_answer_tokens[:, i][:, None],
                                clean_logits=clean_logits, 
                                corrupted_logits=corrupted_logits,
                                word_idx=word_idx,
                                iteration=i,
                                denoising=True)
        act_patch_attn_head_all_pos_every_iteration[i] = \
            patching.get_act_patch_attn_head_all_pos_every(model, corrupted_tokens, clean_cache, compute_score_aux)

        # Unless we are at the last iteration, update the prompts
        if i < clean_answer_tokens.shape[-1] - 1:
            clean_tokens[torch.arange(batch_size), word_idx["END"][:batch_size] + i + 1] = clean_answer_tokens[:, i]
            corrupted_tokens[torch.arange(batch_size), word_idx["END"][:batch_size] + i + 1] = corrupted_answer_tokens[:, i]
            # clean_tokens = torch.cat([clean_tokens, clean_answer_tokens[:, i][:, None]], dim=-1)
            # corrupted_tokens = torch.cat([corrupted_tokens, corrupted_answer_tokens[:, i][:, None]], dim=-1)
    
    return act_patch_attn_head_all_pos_every_iteration


def display_attn_patterns(model: HookedTransformer, tokens, word_idx, important_heads, sample=0):
    """
    Display the attention patterns for the `important_heads` across every iteration.
    If a batch larger than 1 is given, we show only the attn pattern of the first sentence.
    - `tokens`: Tensor[batch, seq_len]
    - `answer_tokens`: Tensor[batch, 3]
    - `important_heads`: List of the form [(layer, head), ...] indicating
    which heads do we want to visualize
    - `sample`: int
        Which sentence of the batch do we want to visualize
    """

    i_end = word_idx["END"][sample] + 3
    _, cache = model.run_with_cache(tokens)
    # Get all their attention patterns
    attn_patterns_for_important_heads = torch.stack([
        cache["pattern", layer][:, head][sample]
        for layer, head in important_heads
    ])

    # Display results
    display(HTML(f"<b>Attention patterns of important heads</b>"))
    display(cv.attention.attention_patterns(
        attention = attn_patterns_for_important_heads[:, :i_end, :i_end],
        tokens = model.to_str_tokens(tokens[sample, :i_end]),
        attention_head_names = [f"{layer}.{head}" for layer, head in important_heads],
    ))

def display_attn_patterns_cache(cache, str_tokens, word_idx, important_heads, sample=0):
    """
    Display the attention patterns for the `important_heads` across every iteration.
    If a batch larger than 1 is given, we show only the attn pattern of the first sentence.
    - `tokens`: Tensor[batch, seq_len]
    - `answer_tokens`: Tensor[batch, 3]
    - `important_heads`: List of the form [(layer, head), ...] indicating
    which heads do we want to visualize
    - `sample`: int
    Which sentence of the batch do we want to visualize
    """
    i_end = word_idx["END"][sample] + 3
    # Get all their attention patterns
    attn_patterns_for_important_heads = torch.stack([
    cache["pattern", layer][:, head][sample]
    for layer, head in important_heads
    ])

    # Display results
    display(HTML(f"<b>Attention patterns of important heads</b>"))
    display(cv.attention.attention_patterns(
    attention = attn_patterns_for_important_heads[:, :i_end, :i_end],
    tokens = str_tokens,
    attention_head_names = [f"{layer}.{head}" for layer, head in important_heads],
    ))


def compute_copying_scores(model: HookedTransformer, predecing_space=True):
    """
    Computes copying score by taking every possible capital letter token | X|, 
    embedding and passing through MLP0 and then passing it through every possible 
    OV circuit, then checking if the initial token is among the top 5 predictions

    Returns a tensor of shape (layer, heads) containing the copying score for every
    head in the model
    """

    copying_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.float32, device=model.cfg.device)

    first_cap = [f" {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    first_cap_token = model.to_tokens(first_cap, prepend_bos=False)

    # Define components from our model
    embed = model.embed
    mlp0 = model.blocks[0].mlp
    ln0 = model.blocks[0].ln2
    unembed = model.unembed
    ln_final = model.ln_final

    # Embed
    embedding = embed(first_cap_token)
    # Pass through MLP0
    # Get residual stream after applying MLP
    resid_after_mlp0 = embedding + mlp0(ln0(embedding))
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            # Pass through the OV circuit of head
            W_OV = model.W_V[layer, head] @ model.W_O[layer, head]
            resid_after_OV = resid_after_mlp0 @ W_OV
            ov_logits = unembed(ln_final(resid_after_OV)).squeeze()
            top_pred_tokens = torch.topk(ov_logits, k=5).indices
            #print(f"Top 5 logits written by OV circuit of head {layer}.{head} when fully attending to |{ first_cap}|:\n\t {model.to_str_tokens(top_pred_tokens)}")
            if predecing_space:
                copying_scores[layer, head] = (first_cap_token == top_pred_tokens).any(-1).float().mean()
            else:
                first_cap_no_space = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
                first_cap_token_no_space = model.to_tokens(first_cap_no_space, prepend_bos=False)
                copying_scores[layer, head] = (first_cap_token_no_space == top_pred_tokens).any(-1).float().mean()

    return copying_scores

############
# PATCHING #
############

def patch_or_freeze_head_vectors(
	orig_head_vector,
	hook, 
	new_cache: ActivationCache,
	orig_cache: ActivationCache,
	head_to_patch, 
):
	'''
	This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
	to their values in orig_cache), except for head_to_patch (if it's in this layer) which
	we patch with the value from new_cache.

	head_to_patch: tuple of (layer, head)
		we can use hook.layer() to check if the head to patch is in this layer
	'''
	# Setting using ..., otherwise changing orig_head_vector will edit cache value too
	orig_head_vector[...] = orig_cache[hook.name][...]
	if head_to_patch[0] == hook.layer():
		orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
	return orig_head_vector


def patch_or_freeze_multi_head_vectors(
    orig_head_vector,
    hook, 
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    heads_to_patch, 
):
    '''
    Performs similar to `patch_or_freeze_head_vectors`, but enables patching multiple heads

    heads_to_patch: List of [(layer1, head1), (layer2, head2), ...]
        we can use hook.layer() to check if the head to patch is in this layer
    '''
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    for head_to_patch in heads_to_patch:
        if head_to_patch[0] == hook.layer():
            orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer, patching_metric, orig_tokens, new_cache, orig_cache,
):
    # SOLUTION
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    '''
    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=model.cfg.device, dtype=torch.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name
    z_name_filter = lambda name: name.endswith("z")

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Looping over every possible sender head (the receiver is always the final resid_post)
    # Note use of itertools (gives us a smoother progress bar)
    for (sender_layer, sender_head) in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):

        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache, 
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn)

        _, patched_cache = model.run_with_cache(
            orig_tokens, 
            names_filter=resid_post_name_filter, 
            return_type=None
        )
        model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == {resid_post_hook_name}

        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        # Save the results
        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results


def get_path_patch_head_to_final_resid_post_iteration(
    model: HookedTransformer, patching_metric, orig_tokens, new_cache, orig_cache, answer_tokens, orig_logits, new_logits, word_idx
    ):
    """
    Applies `get_path_patch_head_to_final_resid_post` for every iteration
    """

    path_patch_head_to_final_resid_post_iteration = torch.zeros(
        (3, model.cfg.n_layers, model.cfg.n_heads), 
        dtype=torch.float32, device=model.cfg.device)

    for i in range(3):
        compute_score_path_patching = partial(patching_metric, correct_answer_token=answer_tokens[:, i][:, None], clean_logits=orig_logits, 
                                        corrupted_logits=new_logits, word_idx=word_idx, iteration=i, denoising=False)
        path_patch_head_to_final_resid_post_iteration[i] = get_path_patch_head_to_final_resid_post(model, compute_score_path_patching, orig_tokens, new_cache, orig_cache)

    return path_patch_head_to_final_resid_post_iteration

def patch_head_input(
    orig_activation, hook: HookPoint,
    patched_cache: ActivationCache,
    head_list,
):
    '''
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    '''
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation


def get_path_patch_head_to_heads(
    receiver_heads,
    receiver_input: str,
    model: HookedTransformer,
    patching_metric,
    orig_tokens,
    new_cache,
    orig_cache,
):
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = input to a later head (or set of heads)

    The receiver node is specified by receiver_heads and receiver_input.
    Example (for S-inhibition path patching the queries):
        receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
        receiver_input = "v"

    Returns:
        tensor of metric values for every possible sender head
    '''
    # SOLUTION
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    results = torch.zeros(max(receiver_layers), model.cfg.n_heads, device=model.cfg.device, dtype=torch.float32)

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we 
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")

    # Note, the sender layer will always be before the final receiver layer, otherwise there will
    # be no causal effect from sender -> receiver. So we only need to loop this far.
    for (sender_layer, sender_head) in tqdm(list(itertools.product(
        range(max(receiver_layers)),
        range(model.cfg.n_heads)
    ))):

        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache, 
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn) #, level=1)

        _, patched_cache = model.run_with_cache(
            orig_tokens, 
            names_filter=receiver_hook_names_filter,  
            return_type=None
        )
        model.reset_hooks(including_permanent=True)
        assert set(patched_cache.keys()) == set(receiver_hook_names)

        # ========== Step 3 ==========
        # Run on x_orig, patching in the receiver node(s) from the previously cached value

        hook_fn = partial(
            patch_head_input, 
            patched_cache=patched_cache, 
            head_list=receiver_heads,
        )
        patched_logits = model.run_with_hooks(
            orig_tokens,
            fwd_hooks = [(receiver_hook_names_filter, hook_fn)], 
            return_type="logits"
        )

        # Save the results
        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results


def get_path_patch_head_to_heads_iteration(
    receiver_heads,
    receiver_input: str,
    model: HookedTransformer,
    patching_metric,
    orig_tokens,
    new_cache,
    orig_cache,
    answer_tokens, 
    orig_logits, 
    new_logits, 
    word_idx   
):
    """
    Obtains the results of the function `get_path_patch_head_to_heads` 
    for every iteration.
    """
    receiver_layers = set(next(zip(*receiver_heads)))
    path_patch_head_to_heads_iteration = torch.zeros(
        (3, max(receiver_layers), model.cfg.n_heads), 
        dtype=torch.float32, device=model.cfg.device)

    for i in range(3):
        compute_score_path_patching = partial(patching_metric, correct_answer_token=answer_tokens[:, i][:, None], clean_logits=orig_logits, 
                                        corrupted_logits=new_logits, word_idx=word_idx, iteration=i, denoising=False)
        path_patch_head_to_heads_iteration[i] = get_path_patch_head_to_heads(    
                                                    receiver_heads,
                                                    receiver_input,
                                                    model,
                                                    compute_score_path_patching,
                                                    orig_tokens,
                                                    new_cache,
                                                    orig_cache)
    
    return path_patch_head_to_heads_iteration


def patch_multiple_heads(
        sender_heads,
        receiver_heads,
        receiver_input: str,
        model: HookedTransformer,
        orig_tokens,
        new_cache,
        orig_cache,
    ):
    """
    Performs path patching from `sender_heads` to `receiver_heads` and returns the `patched_cache`
    """
    model.reset_hooks()

    assert receiver_input in ("k", "q", "v")
    receiver_layers = set(next(zip(*receiver_heads)))
    receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we 
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")


    # ========== Step 2 ==========
    # Run on x_orig, with sender head patched from x_new, every other head frozen
    hook_fn = partial(
        patch_or_freeze_multi_head_vectors,
        new_cache=new_cache, 
        orig_cache=orig_cache,
        heads_to_patch=sender_heads,
    )
    model.add_hook(z_name_filter, hook_fn) #, level=1)

    _, patched_cache = model.run_with_cache(
        orig_tokens, 
        names_filter=receiver_hook_names_filter,  
        return_type=None
    )
    model.reset_hooks(including_permanent=True)
    assert set(patched_cache.keys()) == set(receiver_hook_names)

    # ========== Step 3 ==========
    # Run on x_orig, patching in the receiver node(s) from the previously cached value

    hook_fn = partial(
        patch_head_input, 
        patched_cache=patched_cache, 
        head_list=receiver_heads,
    )
    model.add_hook(receiver_hook_names_filter, hook_fn)
    _, patched_cache = model.run_with_cache(orig_tokens)
    model.reset_hooks(including_permanent=True)

    return patched_cache