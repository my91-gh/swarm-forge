"""Altman Z-Score: financial distress detection for public companies."""

from dataclasses import dataclass


@dataclass
class AltmanInputs:
    working_capital: float
    total_assets: float
    retained_earnings: float
    ebit: float
    market_cap: float
    total_liabilities: float
    sales: float


@dataclass
class AltmanResult:
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    z_score: float
    zone: str  # "safe" | "grey" | "distress"


def compute_z_score(inp: AltmanInputs) -> AltmanResult:
    x1 = inp.working_capital / inp.total_assets
    x2 = inp.retained_earnings / inp.total_assets
    x3 = inp.ebit / inp.total_assets
    x4 = inp.market_cap / inp.total_liabilities
    x5 = inp.sales / inp.total_assets
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    zone = "safe" if z > 2.99 else ("grey" if z >= 1.23 else "distress")
    return AltmanResult(
        x1=round(x1, 4), x2=round(x2, 4), x3=round(x3, 4),
        x4=round(x4, 4), x5=round(x5, 4), z_score=round(z, 3), zone=zone,
    )
