"""Two-stage DCF engine with dynamic WACC and sensitivity analysis."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DCFInputs:
    base_fcf: float
    growth_rates: list[float]  # 5 years explicit
    terminal_growth: float     # max 0.03
    wacc: float
    net_debt: float
    shares_outstanding: float


@dataclass
class DCFResult:
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    sensitivity_table: dict


def run_dcf(inputs: DCFInputs) -> DCFResult:
    assert inputs.terminal_growth <= 0.03, "Terminal growth must not exceed 3%"
    assert len(inputs.growth_rates) == 5, "Provide exactly 5 explicit growth rates"

    fcf = inputs.base_fcf
    pv_fcfs = []
    for i, g in enumerate(inputs.growth_rates):
        fcf *= (1 + g)
        pv = fcf / (1 + inputs.wacc) ** (i + 1)
        pv_fcfs.append(pv)

    terminal_value = (fcf * (1 + inputs.terminal_growth)) / (inputs.wacc - inputs.terminal_growth)
    pv_terminal = terminal_value / (1 + inputs.wacc) ** 5
    enterprise_value = sum(pv_fcfs) + pv_terminal
    equity_value = enterprise_value - inputs.net_debt
    iv_per_share = equity_value / inputs.shares_outstanding

    sensitivity = {}
    for wacc_delta in [-0.01, -0.005, 0, 0.005, 0.01]:
        for g_delta in [-0.005, 0, 0.005]:
            adj_wacc = inputs.wacc + wacc_delta
            adj_g = min(inputs.terminal_growth + g_delta, 0.03)
            if adj_wacc <= adj_g:
                sensitivity[(wacc_delta, g_delta)] = None
                continue
            fcf_s = inputs.base_fcf
            pv_s = []
            for i, g in enumerate(inputs.growth_rates):
                fcf_s *= (1 + g)
                pv_s.append(fcf_s / (1 + adj_wacc) ** (i + 1))
            tv_s = (fcf_s * (1 + adj_g)) / (adj_wacc - adj_g)
            ev_s = sum(pv_s) + tv_s / (1 + adj_wacc) ** 5
            sensitivity[(wacc_delta, g_delta)] = (ev_s - inputs.net_debt) / inputs.shares_outstanding

    return DCFResult(
        intrinsic_value_per_share=round(iv_per_share, 2),
        enterprise_value=round(enterprise_value, 0),
        equity_value=round(equity_value, 0),
        sensitivity_table=sensitivity,
    )
