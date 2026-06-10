"""Beneish M-Score: earnings manipulation detection (8-variable model)."""

from dataclasses import dataclass


@dataclass
class BeneishInputs:
    receivables_t: float
    receivables_t1: float
    sales_t: float
    sales_t1: float
    cogs_t: float
    cogs_t1: float
    total_assets_t: float
    total_assets_t1: float
    ppe_t: float
    ppe_t1: float
    depreciation_t: float
    depreciation_t1: float
    sga_t: float
    sga_t1: float
    net_income_t: float
    interest_expense_t: float
    current_assets_t: float
    current_liabilities_t: float
    cash_t: float
    long_term_debt_t: float
    long_term_debt_t1: float
    cfo_t: float


@dataclass
class BeneishResult:
    dsri: float
    gmi: float
    aqi: float
    sgi: float
    depi: float
    sgai: float
    lvgi: float
    tata: float
    m_score: float
    manipulator: bool


def compute_m_score(inp: BeneishInputs) -> BeneishResult:
    dsri = (inp.receivables_t / inp.sales_t) / (inp.receivables_t1 / inp.sales_t1)
    gm_t = (inp.sales_t - inp.cogs_t) / inp.sales_t
    gm_t1 = (inp.sales_t1 - inp.cogs_t1) / inp.sales_t1
    gmi = gm_t1 / gm_t if gm_t != 0 else 1.0
    nca_t = inp.total_assets_t - inp.current_assets_t - inp.ppe_t
    nca_t1 = inp.total_assets_t1 - inp.current_assets_t1 - inp.ppe_t1
    aqi = (nca_t / inp.total_assets_t) / (nca_t1 / inp.total_assets_t1) if nca_t1 != 0 else 1.0
    sgi = inp.sales_t / inp.sales_t1
    depi = (
        (inp.depreciation_t1 / (inp.ppe_t1 + inp.depreciation_t1))
        / (inp.depreciation_t / (inp.ppe_t + inp.depreciation_t))
        if (inp.ppe_t + inp.depreciation_t) != 0 else 1.0
    )
    sgai = (inp.sga_t / inp.sales_t) / (inp.sga_t1 / inp.sales_t1)
    lev_t = (inp.long_term_debt_t + inp.current_liabilities_t) / inp.total_assets_t
    lev_t1 = (inp.long_term_debt_t1 + inp.current_liabilities_t) / inp.total_assets_t1
    lvgi = lev_t / lev_t1 if lev_t1 != 0 else 1.0
    tata = (inp.net_income_t + inp.interest_expense_t - inp.cfo_t) / inp.total_assets_t
    m_score = (
        -4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
        + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    )
    return BeneishResult(
        dsri=round(dsri, 4), gmi=round(gmi, 4), aqi=round(aqi, 4), sgi=round(sgi, 4),
        depi=round(depi, 4), sgai=round(sgai, 4), lvgi=round(lvgi, 4), tata=round(tata, 4),
        m_score=round(m_score, 4), manipulator=m_score > -1.78,
    )
