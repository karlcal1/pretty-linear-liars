"""Testbed 1: the Mess3 hidden Markov model and the pre-trained transformer of Shai et al. (2024).

Conventions
-----------
* ``T[k, s, t]`` is the probability of emitting token ``k`` and moving to hidden state ``t``
  from hidden state ``s``.  A belief row vector ``b`` is updated by ``b @ T[k]`` and renormalized.
* The probe site is the final LayerNorm output at the last token (``ln_final.hook_normalized``).
  The unembedding is linear from there, so ``logits = activation @ W_U + b_U`` exactly.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from pll.paths import MESS3_CHECKPOINT

MIXING_X = 0.05
EMISSION_ALPHA = 0.85
MAX_LENGTH = 8
D_MODEL = 64
PROBE_SITE = "ln_final.hook_normalized"


def transition_tensor(mixing_x=MIXING_X, emission_alpha=EMISSION_ALPHA):
    """Joint emission/transition tensor T[k, s, t] of the Mess3 process."""
    T = np.zeros((3, 3, 3))
    for state in range(3):
        for next_state in range(3):
            if state == next_state:
                move_prob = 1.0 - 2.0 * mixing_x
            else:
                move_prob = mixing_x
            for token in range(3):
                if token == next_state:
                    emit_prob = emission_alpha
                else:
                    emit_prob = (1.0 - emission_alpha) / 2.0
                T[token, state, next_state] = move_prob * emit_prob
    return T


def next_token_distribution(beliefs, T):
    """Bayes-optimal next-token distribution for each belief row: P(k) = sum_s b_s sum_t T[k, s, t]."""
    emission_given_state = T.sum(axis=2)  # [k, s]
    return beliefs @ emission_given_state.T


def enumerate_prefixes(T, max_length=MAX_LENGTH):
    """Every token sequence of length 1..max_length, with its exact belief state and probability.

    Probabilities are divided by ``max_length`` so that they sum to one over the whole enumeration
    (each length is a complete distribution on its own).
    """
    sequences = []
    beliefs = []
    probabilities = []
    uniform_belief = np.ones(3) / 3.0
    for length in range(1, max_length + 1):
        for sequence in itertools.product(range(3), repeat=length):
            unnormalized = uniform_belief.copy()
            for token in sequence:
                unnormalized = unnormalized @ T[token]
            sequence_prob = unnormalized.sum()
            sequences.append(list(sequence))
            beliefs.append(unnormalized / sequence_prob)
            probabilities.append(sequence_prob / max_length)
    return sequences, np.array(beliefs), np.array(probabilities)


def sample_sequences(T, length, n_sequences):
    """Sample sequences from the HMM using torch's global RNG.

    The sampling order matches the paper's runs, so seeding torch the same way gives the same pools.
    """
    joint_by_state = torch.tensor(T, dtype=torch.float32).permute(1, 2, 0)  # [s, t, k]
    state = torch.randint(0, 3, (n_sequences,))
    tokens = []
    for _ in range(length):
        # flattened index over (next_state, token) pairs: index = next_state * 3 + token
        joint = joint_by_state[state].reshape(n_sequences, -1)
        draw = torch.multinomial(joint, 1).squeeze(1)
        tokens.append(draw % 3)
        state = draw // 3
    return torch.stack(tokens, dim=1)


def beliefs_along_sequences(tokens, T):
    """Belief state after each token, for a batch of token sequences. Returns [n, length, 3]."""
    tokens = np.asarray(tokens)
    n_sequences, length = tokens.shape
    belief = np.ones((n_sequences, 3)) / 3.0
    beliefs = np.zeros((n_sequences, length, 3))
    for position in range(length):
        # per-row update b <- b @ T[token]
        belief = np.einsum("ns,nst->nt", belief, T[tokens[:, position]])
        belief = belief / belief.sum(axis=1, keepdims=True)
        beliefs[:, position] = belief
    return beliefs


def load_mess3_transformer(checkpoint_path=MESS3_CHECKPOINT):
    """The 4-layer, width-64 transformer trained by Shai et al. on Mess3 (TransformerLens format)."""
    config = HookedTransformerConfig(
        n_layers=4, d_model=D_MODEL, d_vocab=3, n_ctx=MAX_LENGTH,
        n_heads=4, d_head=16, d_mlp=256, act_fn="relu", normalization_type="LN", device="cpu",
    )
    model = HookedTransformer(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)
    return model


@dataclass
class ExhaustiveDataset:
    """All prefixes of length 1..8 with activations at the probe site (last token)."""
    sequences: list
    beliefs: np.ndarray        # [n, 3]
    probabilities: np.ndarray  # [n], sums to one
    activations: np.ndarray    # [n, 64]

    def weighted_covariance(self):
        """Probability-weighted covariance of the activations."""
        weights = self.probabilities / self.probabilities.sum()
        mean = weights @ self.activations
        centered = self.activations - mean
        return (centered * weights[:, None]).T @ centered


def build_exhaustive_dataset(model, T):
    sequences, beliefs, probabilities = enumerate_prefixes(T)
    lengths = np.array([len(sequence) for sequence in sequences])
    activations = np.zeros((len(sequences), D_MODEL), dtype=np.float32)
    for length in range(1, MAX_LENGTH + 1):
        rows = np.where(lengths == length)[0]
        tokens = torch.tensor([sequences[row] for row in rows], dtype=torch.long)
        for start in range(0, len(rows), 128):
            batch = tokens[start:start + 128]
            with torch.no_grad():
                _, cache = model.run_with_cache(batch)
            activations[rows[start:start + 128]] = cache[PROBE_SITE][:, length - 1, :].numpy()
    return ExhaustiveDataset(sequences, beliefs, probabilities, activations)


def last_token_activations(model, tokens):
    """Probe-site activation at the last position for a batch of full-length sequences."""
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return cache[PROBE_SITE][:, -1, :].numpy()


def output_distribution(model, tokens):
    """Next-token distribution at the last position, no intervention."""
    with torch.no_grad():
        logits = model(tokens)
    return torch.softmax(logits[:, -1], dim=-1).numpy()


def steered_output_distribution(model, tokens, delta):
    """Next-token distribution after adding ``delta`` to the probe-site activation at the last token.

    ``delta`` has shape [64] (same edit for every sequence) or [batch, 64].
    """
    delta_tensor = torch.tensor(np.asarray(delta), dtype=torch.float32)

    def add_delta(activation, hook):
        activation[:, -1] = activation[:, -1] + delta_tensor
        return activation

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(PROBE_SITE, add_delta)])
    return torch.softmax(logits[:, -1], dim=-1).numpy()


def simplex_to_2d(beliefs):
    """Barycentric beliefs to the equilateral triangle with corners s0=(0,0), s1=(1,0), s2=(1/2, sqrt3/2)."""
    corner_1 = np.array([1.0, 0.0])
    corner_2 = np.array([0.5, np.sqrt(3) / 2])
    return beliefs[:, 1:2] * corner_1 + beliefs[:, 2:3] * corner_2


def corner_start_sequences(T, n_pool=10000):
    """Three sampled sequences whose beliefs are nearest to (.8,.1,.1), (.1,.8,.1), (.1,.1,.8)."""
    pool = sample_sequences(T, MAX_LENGTH, n_pool)
    pool_beliefs = beliefs_along_sequences(pool.numpy(), T)[:, -1]
    targets = [np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.8, 0.1]), np.array([0.1, 0.1, 0.8])]
    chosen = [int(np.argmin(((pool_beliefs - target) ** 2).sum(axis=1))) for target in targets]
    return pool[chosen], pool_beliefs[chosen]


def simplex_grid(resolution):
    """Uniform grid of beliefs over the simplex, (resolution+1)(resolution+2)/2 points."""
    grid = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            grid.append([i / resolution, j / resolution, (resolution - i - j) / resolution])
    return np.array(grid)


def load_testbed(seed=0):
    """Seed both RNGs, load the transformer and build the exhaustive dataset.

    Seeding before the model is constructed keeps torch's RNG stream identical to the paper's runs,
    so sequence pools sampled afterwards are the same.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_mess3_transformer()
    T = transition_tensor()
    dataset = build_exhaustive_dataset(model, T)
    return model, T, dataset


def grid_intervention_loss(model, T, steering, start_tokens, start_beliefs, grid):
    """Intervention loss of a steering map over a grid of target beliefs from each start sequence.

    For each start and each target, the activation is shifted by (target - start_belief) @ steering,
    and the steered output is compared with the Bayes-optimal distribution for the target belief.
    Returns the mean squared error (summed over the three tokens, averaged over interventions) and
    the mean root-squared error.
    """
    squared_errors = []
    for start in range(len(start_beliefs)):
        tokens = start_tokens[start:start + 1]
        for target in grid:
            delta = (target - start_beliefs[start]) @ steering
            steered = steered_output_distribution(model, tokens, delta)[0]
            expected = next_token_distribution(target[None], T)[0]
            squared_errors.append(float(((expected - steered) ** 2).sum()))
    squared_errors = np.array(squared_errors)
    return float(squared_errors.mean()), float(np.sqrt(squared_errors).mean())
