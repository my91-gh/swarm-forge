"""AQR-style factor scorer: VALUE, QUALITY, MOMENTUM, LOW_VOL composite."""

from dataclasses import dataclass
import numpy as np

WEIGHTS = {"value": 0.40, "quality": 0.30, "momentum": 0.20, "low_vol": 0.10}


@dataclass
class FactorInputs:
    pb_ratio: float
    pe_ratio: float
    pfcf_ratio: float
    ev_ebitda: float
    sector_pb_median: float
    sector_pe_median: float
    sector_pfcf_median: float
    sector_ev_ebitda_median: float
    roic_current: float
    roic_3y_trend: float
    gross_margin_current: float
    gross_margin_3y_trend: float
    accruals_ratio: float
    debt_to_equity_trend: float
    return_12_1m: float
    eps_revision_3m: float
    sharpe_3y: float
    max_drawdown_5y: float
    beta_3y: float


@dataclass
class FactorResult:
    value_score: float
    quality_score: float
    momentum_score: float
    low_vol_score: float
    composite_score: float


def _score_ratio(value: float, median: float, lower_is_better: bool = True) -> float:
    if median == 0:
        return 50.0
    ratio = value / median
    if lower_is_better:
        return float(np.clip(100 * (2 - ratio), 0, 100))
    return float(np.clip(100 * ratio, 0, 100))


def compute_factor_scores(inp: FactorInputs) -> FactorResult:
    value_score = sum([
        _score_ratio(inp.pb_ratio, inp.sector_pb_median),
        _score_ratio(inp.pe_ratio, inp.sector_pe_median),
        _score_ratio(inp.pfcf_ratio, inp.sector_pfcf_median),
        _score_ratio(inp.ev_ebitda, inp.sector_ev_ebitda_median),
    ]) / 4

    quality_score = sum([
        float(np.clip(inp.roic_current * 3, 0, 100)),
        75 if inp.roic_3y_trend > 0 else 25,
        float(np.clip(inp.gross_margin_current * 150, 0, 100)),
        75 if inp.gross_margin_3y_trend > 0 else 25,
        float(np.clip(100 * (0.1 - inp.accruals_ratio) / 0.1, 0, 100)),
        75 if inp.debt_to_equity_trend < 0 else 25,
    ]) / 6

    momentum_score = sum([
        float(np.clip(50 + inp.return_12_1m * 200, 0, 100)),
        float(np.clip(50 + inp.eps_revision_3m * 250, 0, 100)),
    ]) / 2

    low_vol_score = sum([
        float(np.clip(inp.sharpe_3y * 50, 0, 100)),
        float(np.clip(100 * (1 - inp.max_drawdown_5y / 0.5), 0, 100)),
        float(np.clip(100 * (1.5 - inp.beta_3y) / 1.5, 0, 100)),
    ]) / 3

    composite = (
        WEIGHTS["value"] * value_score + WEIGHTS["quality"] * quality_score
        + WEIGHTS["momentum"] * momentum_score + WEIGHTS["low_vol"] * low_vol_score
    )
    return FactorResult(
        value_score=round(value_score, 1), quality_score=round(quality_score, 1),
        momentum_score=round(momentum_score, 1), low_vol_score=round(low_vol_score, 1),
        composite_score=round(composite, 1),
    )
