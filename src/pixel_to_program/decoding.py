"""Token-level scene grammar and permutation-insensitive extraction metrics."""

from __future__ import annotations

from collections import Counter

from .scene import MAX_SHAPES, Scene


def allowed_program_tokens(prefix: list[str]) -> tuple[str, ...]:
    """Return legal next tokens, including geometry bounds and EOS placement.

    This constrains the executable scene language; it cannot ensure that the
    scene matches the input image. The prefix excludes the decoder's initial BOS.
    """
    if not isinstance(prefix, list) or len(prefix) > 32:
        raise ValueError("grammar prefix exceeds the 32-token decoder budget")
    phase, count, x, y = "start", 0, 0, 0
    for token in prefix:
        allowed = _allowed(phase, count, x, y)
        if token not in allowed:
            raise ValueError(f"invalid generated token {token!r} in phase {phase}")
        if phase == "start":
            phase = "boundary"
        elif phase == "boundary":
            phase = "done" if token == "END" else "color"
        elif phase == "color":
            phase = "x"
        elif phase == "x":
            x, phase = int(token), "y"
        elif phase == "y":
            y, phase = int(token), "width"
        elif phase == "width":
            phase = "height"
        elif phase == "height":
            count, phase = count + 1, "boundary"
        elif phase == "done":
            phase = "finished"
    return _allowed(phase, count, x, y)


def _allowed(phase: str, count: int, x: int, y: int) -> tuple[str, ...]:
    if phase == "start":
        return ("SCENE",)
    if phase == "boundary":
        return ("END", "RECT", "ELLIPSE") if count < MAX_SHAPES else ("END",)
    if phase == "color":
        return ("RED", "BLUE", "GREEN")
    if phase in {"x", "y"}:
        return tuple(str(value) for value in range(32))
    if phase == "width":
        return tuple(str(value) for value in range(1, 33 - x))
    if phase == "height":
        return tuple(str(value) for value in range(1, 33 - y))
    if phase == "done":
        return ("<eos>",)
    return ("<pad>",)


def primitive_metrics(expected: Scene, actual: Scene, heldout_pair: tuple[str, str] | None = None) -> dict:
    """Match complete primitives as multisets, without background-pixel credit."""
    expected_shapes, actual_shapes = Counter(expected.shapes), Counter(actual.shapes)
    correct = sum((expected_shapes & actual_shapes).values())
    truth, predicted = len(expected.shapes), len(actual.shapes)
    precision = correct / predicted if predicted else float(not truth)
    recall = correct / truth if truth else float(not predicted)
    result = {
        "primitive_true_positive": correct,
        "primitive_false_positive": predicted - correct,
        "primitive_false_negative": truth - correct,
        "primitive_precision": precision,
        "primitive_recall": recall,
        "primitive_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    if heldout_pair:
        expected_heldout = Counter(shape for shape in expected.shapes if (shape.kind, shape.color) == heldout_pair)
        actual_heldout = Counter(shape for shape in actual.shapes if (shape.kind, shape.color) == heldout_pair)
        result.update(
            {
                "heldout_expected_count": sum(expected_heldout.values()),
                "heldout_predicted_count": sum(actual_heldout.values()),
                "heldout_exact_matches": sum((expected_heldout & actual_heldout).values()),
                "heldout_pair_present": bool(actual_heldout),
            }
        )
    return result
