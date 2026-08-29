"""Pure helpers for bounded, token-aligned HF observations."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .errors import ProbeHarnessError


POSITION_MODES = frozenset({"decision", "all"})


def validate_position_mode(value: str) -> str:
    """Return a normalized activation-position mode."""

    normalized = value.strip().casefold()
    if normalized not in POSITION_MODES:
        choices = ", ".join(sorted(POSITION_MODES))
        raise ProbeHarnessError(f"positions must be one of: {choices}")
    return normalized


def capture_activation(
    hidden: torch.Tensor,
    *,
    positions: str,
    expected_positions: int,
) -> torch.Tensor:
    """Detach one transformer output as either the decision row or full sequence."""

    mode = validate_position_mode(positions)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
        raise ProbeHarnessError("activation target did not return hidden-state tensors")
    if hidden.ndim >= 3:
        if int(hidden.shape[0]) != 1:
            raise ProbeHarnessError("activation capture requires a single-item batch")
        sequence = hidden[0]
    else:
        sequence = hidden
    if sequence.ndim != 2 or int(sequence.shape[0]) != expected_positions:
        raise ProbeHarnessError(
            "activation token positions do not align with the exact decision inputs"
        )
    value = sequence[-1] if mode == "decision" else sequence
    return value.detach().float().cpu().contiguous().clone()


def activation_summary(value: torch.Tensor) -> dict[str, object]:
    """Build a compact JSON-safe summary without retaining the source device tensor."""

    activation = value.detach().float().cpu()
    summary: dict[str, object] = {
        "shape": list(activation.shape),
        "dtype": str(activation.dtype).removeprefix("torch."),
        "l2_norm": float(activation.norm().item()),
        "mean": float(activation.mean().item()),
        "std": float(activation.std(unbiased=False).item()),
    }
    if activation.ndim == 2:
        norms = activation.norm(dim=-1)
        summary["position_count"] = int(activation.shape[0])
        summary["position_l2_norm_min"] = float(norms.min().item())
        summary["position_l2_norm_max"] = float(norms.max().item())
        summary["position_l2_norm_last"] = float(norms[-1].item())
    return summary


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def token_position_profile(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    prompt_token_count: int,
    positions: str,
) -> list[dict[str, object]]:
    """Describe exactly the token positions represented by captured activations."""

    mode = validate_position_mode(positions)
    values = [int(token_id) for token_id in token_ids]
    if not values:
        raise ProbeHarnessError("decision inputs contain no token positions")
    if prompt_token_count <= 0 or prompt_token_count > len(values):
        raise ProbeHarnessError("stored prompt token count does not align with decision inputs")
    selected = range(len(values)) if mode == "all" else (len(values) - 1,)
    return [
        {
            "position": position,
            "token_id": values[position],
            "token_text": _decode_token(tokenizer, values[position]),
            "region": "prompt" if position < prompt_token_count else "decision-prefix",
        }
        for position in selected
    ]


def compact_logit_summary(
    tokenizer: Any,
    scores: torch.Tensor,
    *,
    safe_token_id: int,
    unsafe_token_id: int,
    top_k: int = 5,
) -> dict[str, object]:
    """Retain label logits and a bounded top-token view, never the full vocabulary row."""

    values = scores.detach().float().cpu().reshape(-1)
    if not torch.isfinite(values).all():
        raise ProbeHarnessError("subject produced non-finite decision logits")
    count = min(max(int(top_k), 1), int(values.numel()))
    top_values, top_indices = torch.topk(values, count)
    return {
        "safe": {
            "token_id": int(safe_token_id),
            "token_text": _decode_token(tokenizer, int(safe_token_id)),
            "logit": float(values[safe_token_id].item()),
        },
        "unsafe": {
            "token_id": int(unsafe_token_id),
            "token_text": _decode_token(tokenizer, int(unsafe_token_id)),
            "logit": float(values[unsafe_token_id].item()),
        },
        "top_tokens": [
            {
                "rank": rank,
                "token_id": int(token_id),
                "token_text": _decode_token(tokenizer, int(token_id)),
                "logit": float(logit),
            }
            for rank, (token_id, logit) in enumerate(
                zip(top_indices.tolist(), top_values.tolist(), strict=True),
                start=1,
            )
        ],
    }
