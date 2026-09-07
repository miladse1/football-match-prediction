"""Independent Poisson / Dixon-Coles 1X2 probabilities.

Attack and defence ratings plus rho are fit on the train split only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MAX_GOALS = 10
RHO_GRID = np.round(np.linspace(-0.15, 0.15, 13), 4)


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(float(lam), 1e-6)
    logp = -lam + k * np.log(lam) - sum(np.log(np.arange(1, k + 1))) if k else -lam
    return float(np.exp(logp))


def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for i in range(max_goals + 1):
        p_i = _poisson_pmf(i, lam)
        for j in range(max_goals + 1):
            mat[i, j] = p_i * _poisson_pmf(j, mu) * _tau(i, j, lam, mu, rho)
    total = mat.sum()
    if total <= 0:
        mat[:] = 1.0 / mat.size
        return mat
    return mat / total


def matrix_to_1x2(mat: np.ndarray) -> np.ndarray:
    home = float(np.tril(mat, -1).sum())
    away = float(np.triu(mat, 1).sum())
    draw = float(np.trace(mat))
    vec = np.array([away, draw, home], dtype=float)
    s = vec.sum()
    return vec / s if s else np.array([1 / 3, 1 / 3, 1 / 3])


class DixonColesPoisson:
    """Team attack/defence from train averages; rho chosen by train log-likelihood."""

    def __init__(self, max_goals: int = MAX_GOALS):
        self.max_goals = max_goals
        self.classes_ = np.array([0, 1, 2])
        self.league_home_ = 1.5
        self.league_away_ = 1.2
        self.attack_: dict[int, float] = {}
        self.defence_: dict[int, float] = {}
        self.rho_ = 0.0

    def _lambdas(self, home_id: int, away_id: int) -> tuple[float, float]:
        att_h = self.attack_.get(int(home_id), 1.0)
        def_a = self.defence_.get(int(away_id), 1.0)
        att_a = self.attack_.get(int(away_id), 1.0)
        def_h = self.defence_.get(int(home_id), 1.0)
        return self.league_home_ * att_h * def_a, self.league_away_ * att_a * def_h

    def fit(self, home_ids, away_ids, home_goals, away_goals):
        home_ids = np.asarray(home_ids)
        away_ids = np.asarray(away_ids)
        hg = np.asarray(home_goals, dtype=float)
        ag = np.asarray(away_goals, dtype=float)
        self.league_home_ = float(np.mean(hg))
        self.league_away_ = float(np.mean(ag))
        scored: dict[int, list[float]] = defaultdict(list)
        conceded: dict[int, list[float]] = defaultdict(list)
        for hid, aid, h, a in zip(home_ids, away_ids, hg, ag, strict=True):
            scored[int(hid)].append(h / max(self.league_home_, 1e-6))
            conceded[int(hid)].append(a / max(self.league_away_, 1e-6))
            scored[int(aid)].append(a / max(self.league_away_, 1e-6))
            conceded[int(aid)].append(h / max(self.league_home_, 1e-6))
        self.attack_ = {team: float(np.mean(vals)) for team, vals in scored.items()}
        self.defence_ = {team: float(np.mean(vals)) for team, vals in conceded.items()}

        best_rho, best_nll = 0.0, float("inf")
        for rho in RHO_GRID:
            nll = 0.0
            for hid, aid, h, a in zip(home_ids, away_ids, hg, ag, strict=True):
                lam, mu = self._lambdas(hid, aid)
                mat = score_matrix(lam, mu, float(rho), self.max_goals)
                ih, ia = int(min(h, self.max_goals)), int(min(a, self.max_goals))
                nll -= np.log(max(mat[ih, ia], 1e-12))
            if nll < best_nll:
                best_nll, best_rho = nll, float(rho)
        self.rho_ = best_rho
        return self

    def predict_proba(self, home_ids, away_ids) -> np.ndarray:
        rows = []
        for hid, aid in zip(np.asarray(home_ids), np.asarray(away_ids), strict=True):
            lam, mu = self._lambdas(hid, aid)
            rows.append(matrix_to_1x2(score_matrix(lam, mu, self.rho_, self.max_goals)))
        return np.vstack(rows)
