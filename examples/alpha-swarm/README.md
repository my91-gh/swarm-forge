# Alpha Swarm

A multi-agent investment research system built on the [SwarmForge](https://github.com/my91-gh/swarm-forge) architecture. Five specialised agents collaborate to identify high-quality investment opportunities using the best practices of legendary investors.

## Philosophy

| Principle | Source |
|-----------|--------|
| Margin of safety (≥ 30%) | Graham / Buffett |
| Moat analysis (5 dimensions) | Munger / Morningstar |
| Macro cycle positioning | Dalio — All-Weather |
| Downside first | Howard Marks |
| Behavioural bias audit | Kahneman / Thaler |
| Management quality | Buffett / Lynch |

## Agents

```
Screener ──→ Analyst ──→ Quant ──→ Devil's Advocate ──→ Portfolio Manager
                                        │
                                      VETO ──→ Screener (restart)
```

| Agent | Role | Inspired by |
|-------|------|-------------|
| `screener` | Macro context + quantitative filter | Dalio, Lynch |
| `analyst` | Business quality + intrinsic value | Buffett, Munger, Graham |
| `quant` | Factor scoring + fraud/distress detection | Greenblatt, Asness, Altman |
| `devil` | Bias audit + bear case + accounting forensics | Marks, Kahneman, Soros |
| `portfolio` | Final decision + portfolio coherence | Lynch, Dalio |

## Quality Gates (hard blocks)

| Gate | Agent | Condition |
|------|-------|-----------|
| Moat minimum | Analyst | Score < 6/10 → REJECT |
| Margin of safety | Analyst | < 30% → HOLD_FOR_DATA |
| Beneish M-Score | Quant | > −1.78 → REJECT (fraud risk) |
| Altman Z-Score | Quant | < 1.23 → REJECT (distress) |
| Factor Score | Quant | < 65/100 → REJECT |
| Unresolved bias | Devil | Any flagged → resolve or VETO |
| Accounting flags | Devil | 2+ HIGH severity → VETO |

## Output

Each pipeline run produces an **Investment Recommendation Card** in `./output/<ticker>-recommendation.md`:

```
TICKER          : $MSFT
ACTION          : BUY
ENTRY PRICE     : $380.00
INTRINSIC VALUE : $520.00  (base case)
UPSIDE          : 36.8%
POSITION SIZE   : 4.2%
HOLDING PERIOD  : 2–4 years
CONVICTION      : 7/10
```

## Setup

### Prerequisites
- [SwarmForge](https://github.com/my91-gh/swarm-forge) installed
- `claude` CLI available
- Python 3.11+ with: `yfinance`, `pandas`, `numpy`, `requests`

### Install Python dependencies
```bash
pip install yfinance pandas numpy requests sec-edgar-api
```

### Configure API keys
```bash
export FRED_API_KEY=your_key        # https://fred.stlouisfed.org/docs/api/api_key.html
export FMP_API_KEY=your_key         # https://site.financialmodelingprep.com
```

### Launch the swarm
```bash
# From the examples/alpha-swarm directory
swarm
```

## File Structure

```
alpha-swarm/
├── swarmforge/
│   ├── swarmforge.conf              # Agent topology
│   ├── constitution.prompt          # Entry point
│   ├── constitution/
│   │   ├── philosophy.prompt        # Investment principles
│   │   ├── engineering.prompt       # Data sources + tools
│   │   └── workflow.prompt          # Handoff protocol
│   ├── screener.prompt
│   ├── analyst.prompt
│   ├── quant.prompt
│   ├── devil.prompt
│   └── portfolio.prompt
├── tools/
│   ├── dcf_engine.py                # Two-stage DCF
│   ├── factor_scorer.py             # AQR factor scoring
│   ├── beneish.py                   # M-Score (fraud detection)
│   ├── altman.py                    # Z-Score (distress detection)
│   └── kelly.py                     # Kelly position sizing
├── tmp/                             # Agent memos + handoffs (gitignored)
├── output/                          # Final recommendation cards
└── pending-messages/                # Message queue (gitignored)
```

## Disclaimer

This system is for research and educational purposes. It does not constitute
financial advice. Always conduct your own due diligence before making
investment decisions.
