# Overview and objective

The repository contains a Python tool that is intended to automatically process EBA Transparency Exercise data, compute key prudential risk indicators, and generate a traffic-light risk overview across major European banks. It uses Market Risk data, but it could be extended to Credit Risk data as well.

It was created as a coding example for the Supervision Analyst campaign in DG-OMI.

---

## Requirements

```bash
pip install pandas numpy matplotlib requests openpyxl
```

---

## Data source and input

EBA Transparency Exercise, publicly available at:
[https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-transparency-exercise](https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-transparency-exercise). Click on --> Full Database --> Documents --> Market Risk. You can download, from the sae page, the data dictionaery, which is also needed to map the variables.


## Output of the procedure

- **`eba_risk_summary.csv`** : one row per bank, with computed indicators, traffic-light flags (green / amber / red), and an overall risk tier (Low / Medium / High), sorted by risk score
- **`eba_risk_chart.png`** : a multi-panel chart showing the distribution of each indicator across banks and the flag breakdown
- **`eba_market_risk_email.html`** : HTML file for the email, the code sends the image and a brief summary to my email (bignolipaola@gmail.com). You can substitute it with yours

---

## Risk indicators

| Indicator | Description | Source threshold |
|---|---|---|
| CET1 ratio | Common Equity Tier 1 capital / Risk-Weighted Assets | CRR Article 92 |
| NPL ratio | Non-performing loans / Total loans | EBA NPL Guidelines |
| Leverage ratio | Tier 1 capital / Total exposure | CRR Article 429 |
| CRE share | Commercial real estate loans / Total loans | Supervisory guidance |

Each indicator is flagged:
- 🟢 **Green** : within safe range
- 🟡 **Amber** : approaching threshold
- 🔴 **Red** : at or below regulatory minimum

An overall **risk score** (sum of flag weights) determines the bank's risk tier.





## Usage

1. Download the newest (or the one for the year you want to analyse) EBA Transparency Exercise CSV (market data) and save it as `tr_mrk.csv` in the project folder (it should be downloaded directly with this name)
2. Ensure `USE_LOCAL = True` in the configuration section of `eba_risk_report.py`
3. Run:

```bash
python eba_risk_report.py
```

---

## Project structure

```
eba-bank-risk-monitor/
├── eba_risk_report.py # Main pipeline script
├── ## tr_mrk.csv # Input data (download separately)
├── eba_risk_summary.csv # Output: ranked risk summary (generated)
├── eba_risk_chart.png  # Output: multi-panel chart (generated)
├── eba_market_risk_email.html # Output: html for email compilation
└── README.md
```
