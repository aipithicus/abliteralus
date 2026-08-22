"""Evaluation APIs with lazy imports for optional and heavyweight boundaries."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Evaluator": ("abliteralus.evaluation.evaluator", "Evaluator"),
    "perplexity": ("abliteralus.evaluation.metrics", "perplexity"),
    "accuracy": ("abliteralus.evaluation.metrics", "accuracy"),
    "f1_score_metric": ("abliteralus.evaluation.metrics", "f1_score_metric"),
    "refusal_rate": ("abliteralus.evaluation.advanced_metrics", "refusal_rate"),
    "refusal_rate_with_ci": (
        "abliteralus.evaluation.advanced_metrics",
        "refusal_rate_with_ci",
    ),
    "token_kl_divergence": (
        "abliteralus.evaluation.advanced_metrics",
        "token_kl_divergence",
    ),
    "first_token_kl_divergence": (
        "abliteralus.evaluation.advanced_metrics",
        "first_token_kl_divergence",
    ),
    "effective_rank": ("abliteralus.evaluation.advanced_metrics", "effective_rank"),
    "effective_rank_change": (
        "abliteralus.evaluation.advanced_metrics",
        "effective_rank_change",
    ),
    "activation_cosine_similarity": (
        "abliteralus.evaluation.advanced_metrics",
        "activation_cosine_similarity",
    ),
    "linear_cka": ("abliteralus.evaluation.advanced_metrics", "linear_cka"),
    "refusal_projection_magnitude": (
        "abliteralus.evaluation.advanced_metrics",
        "refusal_projection_magnitude",
    ),
    "AbliterationEvalResult": (
        "abliteralus.evaluation.advanced_metrics",
        "AbliterationEvalResult",
    ),
    "format_eval_report": (
        "abliteralus.evaluation.advanced_metrics",
        "format_eval_report",
    ),
    "random_direction_ablation": (
        "abliteralus.evaluation.baselines",
        "random_direction_ablation",
    ),
    "direction_specificity_test": (
        "abliteralus.evaluation.baselines",
        "direction_specificity_test",
    ),
    "arditi_refusal_rate": ("abliteralus.evaluation.heretic_eval", "arditi_refusal_rate"),
    "harmbench_asr": ("abliteralus.evaluation.heretic_eval", "harmbench_asr"),
    "unload_harmbench_classifier": (
        "abliteralus.evaluation.heretic_eval",
        "unload_harmbench_classifier",
    ),
    "first_token_kl_on_prompts": (
        "abliteralus.evaluation.heretic_eval",
        "first_token_kl_on_prompts",
    ),
    "run_lm_eval": ("abliteralus.evaluation.heretic_eval", "run_lm_eval"),
    "load_jailbreakbench_prompts": (
        "abliteralus.evaluation.heretic_eval",
        "load_jailbreakbench_prompts",
    ),
    "run_full_heretic_eval": (
        "abliteralus.evaluation.heretic_eval",
        "run_full_heretic_eval",
    ),
    "format_comparison_table": (
        "abliteralus.evaluation.heretic_eval",
        "format_comparison_table",
    ),
    "HereticComparisonResult": (
        "abliteralus.evaluation.heretic_eval",
        "HereticComparisonResult",
    ),
    "LM_EVAL_BENCHMARKS": (
        "abliteralus.evaluation.heretic_eval",
        "LM_EVAL_BENCHMARKS",
    ),
    "run_benchmarks": (
        "abliteralus.evaluation.lm_eval_integration",
        "run_benchmarks",
    ),
    "compare_models": (
        "abliteralus.evaluation.lm_eval_integration",
        "compare_models",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
