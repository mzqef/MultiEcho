"""Experimental (0.4): a reference law core for any world.

Response blocks (one shared baseline, or one per unit), the declared two-part codes (N1 with replicate
noise, N2 with maximum-likelihood residual variances) and their compression gain ΔL, law families 0-2 (local
interpolation, dense curve, shared profiles) and the zero-response law, backward recovery with a
conformal set, selection by cross-validated ΔL with an indistinguishability report, law cards, and the
law channel. It is one reference implementation of law plugins, not a protocol requirement: other law
families are other plugins. Every methodological level (folds, ranks, confidence, resamples, code) is a
caller's declaration; nothing here encodes a pass rate or a threshold. NumPy only; importing
`multiecho_framework` does not import this subpackage.
"""
from ..law_cards import REUSE_DIMENSIONS, LawCard
from .backward import BackwardResult, conformal_threshold, fold_groups, recover
from .basis import ProfileBasis
from .block import LawScope, ResponseBlock, Unit
from .card import DiscoverySummary, forward_r2, law_card
from .channel import ProfileTransfer, law_channel, law_transfer
from .code import CodeSpec, DeltaL, NoiseModel, data_gain, delta_l, law_bits
from .families import (
    LAW_CLASSES,
    Anchored,
    DenseCurve,
    FittedLaw,
    LocalInterpolation,
    NotApplicable,
    SharedProfile,
    ZeroLaw,
    candidate_ranks,
    law_from_state,
    law_state,
)
from .select import Candidate, CandidateResult, default_candidates, indistinguishable, select

__all__ = [
    "LAW_CLASSES", "REUSE_DIMENSIONS", "Anchored", "BackwardResult", "ProfileTransfer", "Candidate", "CandidateResult", "CodeSpec", "DeltaL", "DenseCurve", "DiscoverySummary",
    "FittedLaw", "LawCard", "LawScope", "LocalInterpolation", "NoiseModel", "NotApplicable", "ProfileBasis",
    "ResponseBlock", "SharedProfile", "Unit", "ZeroLaw", "candidate_ranks", "conformal_threshold", "data_gain",
    "default_candidates", "delta_l", "fold_groups", "forward_r2", "indistinguishable", "law_bits", "law_card", "law_channel", "law_transfer", "law_from_state", "law_state", "recover", "select",
]
