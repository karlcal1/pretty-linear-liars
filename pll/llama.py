"""Testbed 2: Llama-2-7B probed for the birth year of named people.

Prompt: ``"{name} was born in 19"``.  The probed activation is the residual stream at the last token
of the name (output of decoder layer ``PROBE_LAYER``).  The model's answer is summarized as the
expected decade digit E[d] = sum_d d * P(d) over the ten digit tokens.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from pll.paths import BIRTH_YEAR_FILES, LLAMA_MODEL_ID

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

PROMPT_SUFFIX = " was born in 19"
PROBE_LAYER = 8
N_ENTITIES = 8000
RIDGE_LAMBDA = 1e4          # the intervention-optimal penalty (Figure 2)
LAMBDA_GRID = [10.0 ** exponent for exponent in range(-3, 9)]
BATCH_SIZE = 8


def load_llama(model_id=LLAMA_MODEL_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_id} on {DEVICE} ...", flush=True)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(DEVICE).eval()
    model.requires_grad_(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"loaded in {time.time() - start:.0f}s", flush=True)
    return tokenizer, model


def digit_token_ids(tokenizer):
    """Token id of each digit 0-9 as it appears after '19'."""
    return [tokenizer(f"19{digit}", add_special_tokens=False).input_ids[-1] for digit in "0123456789"]


def load_birth_year_entities(n_entities=N_ENTITIES, seed=0, files=BIRTH_YEAR_FILES):
    """Unique people born 1900-1999, shuffled with a fixed seed. Returns names and birth years."""
    rows = []
    for path in files:
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                year = int(record["value"])
                if 1900 <= year <= 1999:
                    rows.append((record["entity_label"], year))
    seen = set()
    unique_rows = []
    for name, year in rows:
        if name not in seen:
            seen.add(name)
            unique_rows.append((name, year))
    random.Random(seed).shuffle(unique_rows)
    unique_rows = unique_rows[:n_entities]
    names = [row[0] for row in unique_rows]
    years = np.array([row[1] for row in unique_rows], dtype=float)
    return names, years


def decades_of(years):
    return (years % 100 // 10).astype(int)


def standardize(years):
    return (years - years.mean()) / years.std()


def name_final_indices(tokenizer, names):
    """Position of the last name token in the prompt (the BOS token sits at position 0)."""
    return [len(tokenizer(name).input_ids) - 1 for name in names]


def _patch_hook(delta, positions):
    """Forward hook adding delta[b] to the hidden state of sequence b at positions[b]."""

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        for b in range(hidden.shape[0]):
            # add in float32, then cast back to the model dtype (bfloat16)
            hidden[b, positions[b], :] = (hidden[b, positions[b], :].float() + delta[b]).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    return hook


def cache_activations(model, tokenizer, names, layer, batch_size=BATCH_SIZE):
    """Residual-stream activation at the last name token, for every prompt. Returns float32 [n, d]."""
    positions = name_final_indices(tokenizer, names)
    captured = {}

    def capture(module, inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    activations = []
    for start in range(0, len(names), batch_size):
        prompts = [name + PROMPT_SUFFIX for name in names[start:start + batch_size]]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        handle = model.model.layers[layer].register_forward_hook(capture)
        with torch.no_grad():
            model(encoded.input_ids.to(DEVICE))
        handle.remove()
        hidden = captured["hidden"].float()
        for b, position in enumerate(positions[start:start + batch_size]):
            activations.append(hidden[b, position, :].cpu().numpy())
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return np.stack(activations), positions


def expected_decade(model, tokenizer, names, digit_ids, layer, delta=None, positions=None,
                    batch_size=BATCH_SIZE):
    """E[decade digit] at the last prompt token, optionally after patching ``delta`` [n, d] into the
    residual stream at ``positions`` (one per prompt) at the output of ``layer``."""
    decade_values = torch.arange(10, device=DEVICE, dtype=torch.float32)
    outputs = []
    for start in range(0, len(names), batch_size):
        prompts = [name + PROMPT_SUFFIX for name in names[start:start + batch_size]]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(DEVICE)
        attention_mask = encoded.attention_mask.to(DEVICE)
        handle = None
        if delta is not None:
            batch_delta = torch.tensor(delta[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            handle = model.model.layers[layer].register_forward_hook(
                _patch_hook(batch_delta, positions[start:start + batch_size]))
        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits.float()
        if handle is not None:
            handle.remove()
        last_positions = attention_mask.sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.shape[0]), last_positions, :]
        digit_probs = torch.softmax(last_logits[:, digit_ids], dim=-1)
        outputs.append((digit_probs * decade_values).sum(dim=1).cpu().numpy())
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return np.concatenate(outputs)


def interchange_delta(activations, direction, base_idx, source_idx):
    """Natural-magnitude interchange along a unit direction: delta = (v.a_src - v.a_base) v."""
    swap = (activations[source_idx] - activations[base_idx]) @ direction
    return swap[:, None] * direction[None, :]


def train_das_direction(model, tokenizer, names, decades, activations, positions, layer, digit_ids,
                        seed, steps=500, batch_size=12, learning_rate=1e-2, projector=None):
    """Distributed Alignment Search for a single unit direction.

    Random init; Adam on the MSE between the expected decade after a base<-source interchange and the
    source's decade.  ``projector`` (optional, [d, k] orthonormal columns) confines the direction to a
    subspace, used for the restricted-DAS control.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = activations.shape[1]
    raw_direction = nn.Parameter(torch.randn(d, device=DEVICE) * 0.02)
    optimizer = torch.optim.Adam([raw_direction], lr=learning_rate)
    activations_t = torch.tensor(activations, dtype=torch.float32, device=DEVICE)
    decades_t = torch.tensor(decades, dtype=torch.float32, device=DEVICE)
    decade_values = torch.arange(10, device=DEVICE, dtype=torch.float32)
    n = len(names)

    def current_unit_direction(vector):
        if projector is not None:
            vector = projector @ (projector.T @ vector)
        return vector / vector.norm()

    for _ in range(steps):
        base_idx = np.random.randint(0, n, batch_size)
        source_idx = np.random.randint(0, n, batch_size)
        direction = current_unit_direction(raw_direction)
        swap = (activations_t[source_idx] - activations_t[base_idx]) @ direction
        delta = swap[:, None] * direction[None, :]
        prompts = [names[b] + PROMPT_SUFFIX for b in base_idx]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded.input_ids.to(DEVICE)
        attention_mask = encoded.attention_mask.to(DEVICE)
        handle = model.model.layers[layer].register_forward_hook(
            _patch_hook(delta, [positions[b] for b in base_idx]))
        logits = model(input_ids, attention_mask=attention_mask).logits
        handle.remove()
        last_positions = attention_mask.sum(dim=1) - 1
        last_logits = logits[torch.arange(batch_size), last_positions, :]
        predicted_decade = (torch.softmax(last_logits[:, digit_ids], dim=-1) * decade_values).sum(dim=1)
        loss = ((predicted_decade - decades_t[source_idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return current_unit_direction(raw_direction.detach()).cpu().numpy()


def interchange_metrics(intended, actual):
    """Correlation, slope and MSE of actual output shift against the intended (ground-truth) shift."""
    correlation = float(np.corrcoef(intended, actual)[0, 1])
    slope = float(np.polyfit(intended, actual, 1)[0])
    mse = float(np.mean((actual - intended) ** 2))
    return {"corr": correlation, "slope": slope, "interv_mse": mse}
