"""Abliteralus — Master Ablation Suite for HuggingFace transformers."""

__version__ = "0.1.2"

# Lazy imports for the main pipeline classes
__all__ = [
    "AbliterationPipeline",
    "InformedAbliterationPipeline",
    "set_seed",
]


def __getattr__(name):
    if name == "AbliterationPipeline":
        from abliteralus.abliterate import AbliterationPipeline
        return AbliterationPipeline
    if name == "InformedAbliterationPipeline":
        from abliteralus.informed_pipeline import InformedAbliterationPipeline
        return InformedAbliterationPipeline
    if name == "set_seed":
        from abliteralus.reproducibility import set_seed
        return set_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
