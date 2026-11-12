"""Synthetic ErrP feedback with a controlled decoder quality (population AUROC).

Equal-variance binormal decoder model. For a labelled segment with simulated label y
(1 = error, 0 = non-error) a decoder margin is drawn as

    z = mu_y + eps,   eps ~ N(0, 1),   mu_1 = +d'/2,  mu_0 = -d'/2,
    d' = sqrt(2) * Phi^{-1}(AUROC).

Then P(z_error > z_nonerror) = Phi(d' / sqrt(2)) = AUROC exactly (population value).
The margin is converted to a Bayes-calibrated error probability under a fixed base rate pi:

    p_error = P(y = 1 | z) = sigmoid(logit(pi) + d' * z)

(the log-likelihood ratio of the two unit-variance normals is exactly d' * z).

Base rate. pi is FIXED at 0.35 (default), not the on-policy empirical rate: (1) the real
frozen EEG decoders replayed by ``rl.feedback.FeedbackLibrary`` are calibrated to the EEG
training prevalence (about 0.35-0.36 machine-error trials, reports/eeg_results.md), so a
fixed 0.35 places the synthetic decoders on the same probability scale as the real ones and
the frontier is directly comparable with the eegnet/cbramod points; (2) the error rate of
labelled segments is non-stationary (it falls as the policy improves) and differs across
tasks and conditions, so calibrating to it would make the feedback mapping depend on the
learner itself. Consequence (shared with the real decoders): an uninformative decoder
(AUROC -> 0.5) outputs p = 0.35 for every segment, so w*(0.5 - p) is a constant +0.15 bonus
per labelled (outside-deadband) segment. AUROC = 1.0 is handled by the existing ``ideal``
condition (p = label), not by this class.

Determinism and pairing. eps for an event is a deterministic function of
(feedback seed, event index) only, via SHA-256, so (i) runs are reproducible, and (ii) the
same event index uses the same eps at every AUROC level and either label (common random
numbers across the frontier). As with the real replay, diverging policies produce different
label sequences, so whole on-policy feedback streams are not identical across conditions.

The class duck-types the parts of ``FeedbackLibrary`` used by ``rl.environment.FeedbackReward``
(``profile['splits'][split]`` and ``sample``), so the existing, tested reward wrapper is reused
unchanged with condition ``eegnet`` and subject = the synthetic source name.
"""
from __future__ import annotations

import hashlib
import json
import math
from statistics import NormalDist

_N = NormalDist()
AUROC_LEVELS = (0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0)
DEFAULT_BASE_RATE = 0.35


def dprime_for_auroc(auroc):
    if not 0.5 <= auroc < 1.0:
        raise ValueError("Synthetic decoder AUROC must be in [0.5, 1); use the ideal condition for 1.0")
    return math.sqrt(2.0) * _N.inv_cdf(auroc)


def calibrated_p(z, dprime, base_rate):
    logit = math.log(base_rate / (1.0 - base_rate)) + dprime * z
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


def keyed_normal(seed, event_index):
    """Deterministic N(0,1) draw from the sampling key (seed, event index)."""
    key = json.dumps([int(seed), "synthetic-errp", int(event_index)]).encode()
    bits = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11  # 53 bits
    u = (bits + 0.5) / float(1 << 53)  # strictly inside (0, 1)
    return _N.inv_cdf(u)


class SyntheticFeedbackSource:
    """Controlled-AUROC calibrated decoder; drop-in for FeedbackLibrary in FeedbackReward."""

    split = "synthetic"

    def __init__(self, auroc, base_rate=DEFAULT_BASE_RATE):
        if not 0.0 < base_rate < 1.0:
            raise ValueError("base_rate must be in (0, 1)")
        self.auroc, self.base_rate = float(auroc), float(base_rate)
        self.dprime = dprime_for_auroc(self.auroc)
        self.name = f"synthetic_auroc{self.auroc:.2f}"
        self.profile = {"splits": {self.split: [self.name]}}
        self.table_sha256 = None

    def margin(self, label, seed, event_index):
        mu = self.dprime / 2.0 if label == 1 else -self.dprime / 2.0
        return mu + keyed_normal(seed, event_index)

    def p_error(self, label, seed, event_index):
        return calibrated_p(self.margin(label, seed, event_index), self.dprime, self.base_rate)

    def sample(self, model, subject, label, seed, event_index, independent=False):
        if subject != self.name or independent:
            raise ValueError("Synthetic source supports only its own label-conditioned draws")
        if label not in (0, 1) or event_index < 0:
            raise ValueError("Invalid sampling key")
        p = self.p_error(label, seed, event_index)
        return {"p_error": p, "trial_id": f"syn{int(event_index)}", "label": int(label)}
