"""Model selection by cross-validated ΔL on discovery; validation only reports.

The zero-response law is always a candidate; if no candidate beats it, the reading is
"no law with ΔL > 0 found under plugin set vN" — never "the world has no law" (principle 10).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .backward import fold_groups
from .block import ResponseBlock
from .code import NoiseModel, data_gain, law_bits
from .families import (
    DenseCurve,
    FittedLaw,
    LocalInterpolation,
    NotApplicable,
    SharedProfile,
    ZeroLaw,
    candidate_ranks,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    fit: Callable[[ResponseBlock, NoiseModel], FittedLaw]


@dataclass(frozen=True)
class CandidateResult:
    name: str
    cv_delta_l: float | None
    law: FittedLaw | None
    reason: str = ""
    per_context: dict = field(default_factory=dict, repr=False)   # held-out-fold data gain per context


def default_candidates(block: ResponseBlock, *, max_rank: int = 6) -> tuple[Candidate, ...]:
    candidates = [Candidate("zero", ZeroLaw.fit), Candidate("family0:pl-axis", LocalInterpolation.fit),
                  Candidate("family1:p-spline", DenseCurve.fit)]
    for rank in candidate_ranks(block, max_rank):
        candidates.append(Candidate(f"family2:K{rank}",
                                    lambda data, noise, rank=rank: SharedProfile.fit(data, noise, rank=rank)))
        candidates.append(Candidate(f"family2:K{rank}:gain", lambda data, noise, rank=rank: SharedProfile.fit(
            data, noise, rank=rank, design="gain")))
    return tuple(candidates)


def select(block: ResponseBlock, noise: NoiseModel, candidates: tuple[Candidate, ...], *,
           folds: int) -> tuple[CandidateResult, ...]:
    """Rank candidates by cross-validated ΔL; the choice costs log2(number of candidates) bits.

    Structural hyperparameters (rank, smoothing) are chosen once on all discovery units and then
    held fixed in the context folds; the CV-ΔL is therefore slightly optimistic for those choices.
    """
    names = [candidate.name for candidate in candidates]
    if "zero" not in names or len(set(names)) != len(names):
        raise ValueError("Candidates are uniquely named and include the zero-response law")
    choice_bits = math.log2(len(candidates))
    groups = fold_groups(block.contexts, folds)
    results = []
    for candidate in candidates:
        try:
            law = candidate.fit(block, noise)
            if candidate.name != "zero":
                law = replace(law, struct_bits=law.struct_bits + choice_bits)
            per_context: dict[str, float] = {}
            for held in groups:
                fold_law = law.refit(block.subset(set(block.contexts) - set(held)), noise)
                per_context.update(data_gain(fold_law, block.subset(held), noise)[0])
            results.append(CandidateResult(candidate.name, sum(per_context.values()) - law_bits(law), law,
                                           per_context=per_context))
        except NotApplicable as reason:
            results.append(CandidateResult(candidate.name, None, None, str(reason)))
        except np.linalg.LinAlgError as reason:
            # A numerically singular fit on these data is recorded like an inapplicable family, not ranked.
            results.append(CandidateResult(candidate.name, None, None, f"numerically singular: {reason}"))
    return tuple(sorted(results, key=lambda item: (item.cv_delta_l is None, -(item.cv_delta_l or 0.0), item.name)))


def indistinguishable(ranked: tuple[CandidateResult, ...], *, confidence: float, resamples: int,
                      seed: int) -> tuple[str, ...]:
    """Candidates the data cannot tell apart from the selected one (law recovery is relative to the
    experiment catalog and noise, arXiv 2610.06877).

    For each fitted candidate, the paired difference of cross-validated ΔL to the selected law is
    bootstrapped over contexts (law bits are fixed); a candidate whose interval reaches zero at the
    declared confidence is reported as indistinguishable. Confidence and budget are the caller's.
    """
    best = ranked[0]
    if best.cv_delta_l is None:
        return ()
    contexts = sorted(best.per_context)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(contexts), size=(resamples, len(contexts)))
    tail = (1 - confidence) / 2
    result = []
    for item in ranked[1:]:
        if item.cv_delta_l is None or sorted(item.per_context) != contexts:
            continue
        difference = np.array([best.per_context[c] - item.per_context[c] for c in contexts])
        bits = (best.cv_delta_l - sum(best.per_context.values())) - (item.cv_delta_l - sum(item.per_context.values()))
        lower = float(np.quantile(difference[draws].sum(axis=1) + bits, tail))
        if lower <= 0:
            result.append(item.name)
    return tuple(result)
