"""Abliteralus — Master Ablation Suite for HuggingFace transformers."""

__version__ = "0.1.2"

# Lazy imports for the main pipeline classes
__all__ = [
    "AbliterationPipeline",
    "InformedAbliterationPipeline",
    "set_seed",
    "run_sweep",
    "SweepConfig",
    "SweepResult",
    "save_contribution",
    "load_contributions",
    "aggregate_results",
    "TourneyRunner",
    "TourneyResult",
    "get_adaptive_recommendation",
    "AdaptiveRecommendation",
    "RemoteRunner",
    "RemoteConfig",
    "Watchtower",
    "get_watchtower",
    "AutoObliterator",
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
    if name == "run_sweep":
        from abliteralus.sweep import run_sweep
        return run_sweep
    if name == "SweepConfig":
        from abliteralus.sweep import SweepConfig
        return SweepConfig
    if name == "SweepResult":
        from abliteralus.sweep import SweepResult
        return SweepResult
    if name == "save_contribution":
        from abliteralus.community import save_contribution
        return save_contribution
    if name == "load_contributions":
        from abliteralus.community import load_contributions
        return load_contributions
    if name == "aggregate_results":
        from abliteralus.community import aggregate_results
        return aggregate_results
    if name == "TourneyRunner":
        from abliteralus.tourney import TourneyRunner
        return TourneyRunner
    if name == "TourneyResult":
        from abliteralus.tourney import TourneyResult
        return TourneyResult
    if name == "get_adaptive_recommendation":
        from abliteralus.adaptive_defaults import get_adaptive_recommendation
        return get_adaptive_recommendation
    if name == "AdaptiveRecommendation":
        from abliteralus.adaptive_defaults import AdaptiveRecommendation
        return AdaptiveRecommendation
    if name == "RemoteRunner":
        from abliteralus.remote import RemoteRunner
        return RemoteRunner
    if name == "RemoteConfig":
        from abliteralus.remote import RemoteConfig
        return RemoteConfig
    if name == "Watchtower":
        from abliteralus.watchtower import Watchtower
        return Watchtower
    if name == "get_watchtower":
        from abliteralus.watchtower import get_watchtower
        return get_watchtower
    if name == "AutoObliterator":
        from abliteralus.auto_obliterate import AutoObliterator
        return AutoObliterator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
