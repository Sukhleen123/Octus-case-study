# SimFin Metrics Reference

Available metrics by financial statement type. These are discovered by
`src/simfin/metrics_catalog.py` and written to `data/processed/simfin/metrics_catalog.json`.

## Income Statement

| Metric | Description |
|--------|-------------|
| Revenue | Total revenue / net sales |
| Cost of Revenue | Direct costs of goods sold |
| Gross Profit | Revenue - Cost of Revenue |
| Operating Income | Gross Profit - Operating Expenses |
| Net Income | Bottom-line profit after all expenses and taxes |
| EPS Diluted | Earnings per share (diluted) |
| EBITDA | Earnings before interest, taxes, depreciation, and amortization |
| Depreciation & Amortization | Non-cash expense for asset usage |
| Interest Expense | Cost of debt financing |
| Income Tax Expense | Taxes paid on earnings |

## Balance Sheet

| Metric | Description |
|--------|-------------|
| Total Assets | All assets owned |
| Total Liabilities | All financial obligations |
| Total Equity | Assets minus liabilities (shareholders' equity) |
| Cash & Equivalents | Liquid assets |
| Total Debt | Short + long term borrowings |
| Net Debt | Total Debt - Cash & Equivalents |

## Cash Flow Statement

| Metric | Description |
|--------|-------------|
| Operating Cash Flow | Cash generated from core business operations |
| Investing Cash Flow | Cash from investment activities (capex, acquisitions) |
| Financing Cash Flow | Cash from debt/equity issuance and repayments |
| Free Cash Flow | Operating Cash Flow - Capital Expenditures |
| Capital Expenditures | Investment in fixed assets |

## SimFin Load Variants

The `simfin` Python package provides `sf.load_income()`, `sf.load_balance()`, `sf.load_cashflow()`
with `variant` parameter:

| Variant | Meaning |
|---------|---------|
| `annual` | Full fiscal year statements |
| `quarterly` | Quarterly statements |
| `ttm` | Trailing twelve months (rolling) |

**Reference:** [SimFin load.py](https://github.com/SimFin/simfin/blob/master/simfin/load.py)
