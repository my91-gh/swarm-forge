"""Kelly Criterion position sizing for investment decisions."""

from dataclasses import dataclass


@dataclass
class KellyInputs:
    current_price: float
    intrinsic_value_base: float
    intrinsic_value_bear: float
    margin_of_safety: float   # decimal, e.g. 0.35
    analyst_conviction: int   # 1–10


@dataclass
class KellyResult:
    p_success: float
    b_ratio: float
    full_kelly: float
    half_kelly: float
    max_position_pct: float
    recommended_pct: float


def compute_kelly(inp: KellyInputs) -> KellyResult:
    p = min(0.75, inp.margin_of_safety * 0.5 + inp.analyst_conviction * 0.03 + 0.25)
    upside = inp.intrinsic_value_base - inp.current_price
    downside = inp.current_price - inp.intrinsic_value_bear
    if downside <= 0:
        downside = inp.current_price * 0.2
    b = upside / downside if downside > 0 else 1.0
    q = 1 - p
    full_kelly = max(0.0, (p * b - q) / b) if b > 0 else 0.0
    half_kelly = full_kelly / 2
    recommended = min(half_kelly, 0.10)
    return KellyResult(
        p_success=round(p, 3), b_ratio=round(b, 3),
        full_kelly=round(full_kelly, 4), half_kelly=round(half_kelly, 4),
        max_position_pct=0.10, recommended_pct=round(recommended, 4),
    )
