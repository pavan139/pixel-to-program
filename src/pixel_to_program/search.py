"""Bounded batched beam search with explicit probability and completion semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence


@dataclass(frozen=True)
class Hypothesis:
    tokens: tuple[int, ...]
    log_probability: float
    finished: bool

    def score(self, length_penalty: float) -> float:
        return self.log_probability / (((5 + max(1, len(self.tokens))) / 6) ** length_penalty)


def beam_search(
    next_log_probabilities: Callable[[list[tuple[int, ...]]], Sequence[Sequence[float]]],
    *,
    eos_id: int,
    beam_size: int = 4,
    max_tokens: int = 32,
    length_penalty: float = 0.0,
    allowed_tokens: Callable[[tuple[int, ...]], Sequence[int]] | None = None,
) -> dict:
    """Search original-model log probabilities; masking does not renormalize them.

    Completed hypotheses occupy beam slots while retained and are never
    expanded. Archive completions that entered the retained beam, so later
    unfinished prefixes cannot permanently discard a known completion. At the
    limit select the best archived completion, or expose an unfinished prefix.
    Length normalization is a heuristic, not a probability. Width one agrees
    with greedy decoding because immediately pruned EOS candidates are not kept.
    """
    if type(beam_size) is not int or not 1 <= beam_size <= 16:
        raise ValueError("beam_size must be an integer in [1, 16]")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 64:
        raise ValueError("max_tokens must be an integer in [1, 64]")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("eos_id must be a nonnegative integer")
    if type(length_penalty) not in (int, float) or not math.isfinite(length_penalty) or not 0 <= length_penalty <= 2:
        raise ValueError("length_penalty must be finite in [0, 2]")
    beams = [Hypothesis((), 0.0, False)]
    completed = []
    dead_end = False
    calls = expansions = 0
    width = None
    for _ in range(max_tokens):
        active = [hypothesis for hypothesis in beams if not hypothesis.finished]
        if not active:
            break
        probabilities = list(next_log_probabilities([hypothesis.tokens for hypothesis in active]))
        calls += 1
        if len(probabilities) != len(active):
            raise ValueError("backend must return one distribution per active prefix")
        candidates = [hypothesis for hypothesis in beams if hypothesis.finished]
        for hypothesis, values in zip(active, probabilities, strict=True):
            values = list(values)
            if not values or any(
                type(value) not in (int, float) or math.isnan(value) or value == math.inf for value in values
            ):
                raise ValueError("log probabilities must be real and cannot contain NaN or positive infinity")
            if width is None:
                width = len(values)
            if len(values) != width or eos_id >= width:
                raise ValueError("backend vocabulary changed or EOS is outside its vocabulary")
            peak = max(values)
            if not math.isfinite(peak):
                raise ValueError("backend has no probability mass")
            log_total = peak + math.log(math.fsum(math.exp(value - peak) for value in values))
            if not math.isclose(log_total, 0.0, abs_tol=2e-5):
                raise ValueError("backend must return normalized full-vocabulary log probabilities")
            allowed = list(allowed_tokens(hypothesis.tokens)) if allowed_tokens else list(range(width))
            if any(type(token) is not int or not 0 <= token < width for token in allowed) or len(set(allowed)) != len(
                allowed
            ):
                raise ValueError("allowed token IDs must be unique integers inside the vocabulary")
            for token in allowed:
                if math.isfinite(values[token]):
                    candidates.append(
                        Hypothesis(
                            hypothesis.tokens + (token,), hypothesis.log_probability + values[token], token == eos_id
                        )
                    )
                    expansions += 1
        if not candidates:
            dead_end = True
            break
        beams = sorted(candidates, key=lambda item: (-item.score(length_penalty), item.tokens))[:beam_size]
        archive = {hypothesis.tokens: hypothesis for hypothesis in completed}
        archive.update({hypothesis.tokens: hypothesis for hypothesis in beams if hypothesis.finished})
        completed = sorted(archive.values(), key=lambda item: (-item.score(length_penalty), item.tokens))[:beam_size]
    selected = (completed or beams)[0]
    return {
        "status": "completed"
        if selected.finished
        else "no_legal_continuation"
        if dead_end
        else "token_budget_exhausted",
        "tokens": list(selected.tokens),
        "eos_emitted": selected.finished,
        "log_probability": selected.log_probability,
        "normalized_score": selected.score(length_penalty),
        "length_penalty": length_penalty,
        "backend_calls": calls,
        "candidate_expansions": expansions,
        "beams": [{**asdict(item), "normalized_score": item.score(length_penalty)} for item in beams],
        "completed_hypotheses": [
            {**asdict(item), "normalized_score": item.score(length_penalty)} for item in completed
        ],
    }
