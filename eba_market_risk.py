"""
EBA Transparency Exercise – Market Risk Analyser
-------------------------------------------------
Reads tr_mrk.csv (EBA Transparency Exercise, market risk template),
pivots the long-format data into one row per bank, computes derived
indicators, applies traffic-light flags, and produces a multi-panel
summary chart (PNG) and a ranked summary (CSV).

Data source:
  https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-transparency-exercise
  Download: tr_mrk.csv (Market Risk template)

Data dictionary: SDD.xlsx (same page)

Usage:
  Place tr_mrk.csv in the same directory and run:
      python eba_market_risk.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_FILE   = "tr_mrk.csv"
OUTPUT_CSV   = "eba_market_risk_summary.csv"
OUTPUT_CHART = "eba_market_risk_chart.png"

# Item codes from SDD (TR2024 vintage = 2420xxx; TR2025 = 2520xxx)
# The script detects the vintage automatically from the data.
ITEM_LABELS = {
    "total_rwa":       ("TOTAL RISK EXPOSURE AMOUNT",               [2520401, 2420401]),
    "var_60d":         ("VaR – 60-day avg (mc × VaRavg)",           [2520411, 2420411]),
    "var_prev":        ("VaR – Previous Day (VaRt-1)",              [2520421, 2420421]),
    "svar_60d":        ("SVaR – 60-day avg (ms × SVaRavg)",         [2520431, 2420431]),
    "svar_prev":       ("SVaR – Latest Available (SVaRt-1)",        [2520441, 2420441]),
    "irc_12w":         ("IRC – 12-week Average",                    [2520451, 2420451]),
    "irc_last":        ("IRC – Last Measure",                       [2520461, 2420461]),
    "ctp_12w":         ("CTP All-Price-Risk – 12-week Average",     [2520481, 2420481]),
    "ctp_last":        ("CTP All-Price-Risk – Last Measure",        [2520491, 2420491]),
}

# Derived ratios and their traffic-light thresholds
# var_to_rwa: VaR (60d) as % of market RWA — higher = more model-intensive
# svar_ratio: SVaR / VaR ratio — stress multiplier; elevated > 5x is a flag
THRESHOLDS = {
    # VaR (60d) as % of IMA RWA: typical range 1-3%; warn above 2.5%, breach above 4%
    "var_to_rwa_pct":  {"warn": 2.5,  "breach": 4.0,  "higher_riskier": True},
    # SVaR/VaR ratio: stress multiplier; warn above 5x, breach above 8x
    "svar_var_ratio":  {"warn": 5.0,  "breach": 8.0,  "higher_riskier": True},
    # IRC (12w) as % of IMA RWA: warn above 5%, breach above 10%
    "irc_to_rwa_pct":  {"warn": 5.0,  "breach": 10.0, "higher_riskier": True},
}

COLORS = {"red": "#E05C5C", "amber": "#F5A623", "green": "#4CAF8A", "grey": "#C8D0D8"}
BG_COLOR    = "#F0F4F8"   # page background
PANEL_COLOR = "#FFFFFF"   # panel background
GRID_COLOR  = "#E8ECF0"   # gridlines
ACCENT      = "#2C3E6B"   # title / axis text


# ---------------------------------------------------------------------------
# 2. LOAD & VALIDATE
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load tr_mrk.csv and validate expected columns."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path, sep="\t", low_memory=False)

    required = {"LEI_Code", "NSA", "Period", "Item", "Amount",
                "Portfolio", "MKT_Modprod", "Mkt_risk"}
    missing = required - set(df.columns)
    if missing:
        # Try comma separator fallback
        df = pd.read_csv(path, sep=",", low_memory=False)
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"ERROR: Missing columns: {missing}")

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Item"]   = pd.to_numeric(df["Item"],   errors="coerce").astype("Int64")
    print(f"  {len(df):,} rows | {df['LEI_Code'].nunique()} banks "
          f"| Periods: {sorted(df['Period'].unique())}")
    return df


# ---------------------------------------------------------------------------
# 3. PIVOT TO BANK-LEVEL WIDE FORMAT
# ---------------------------------------------------------------------------

def extract_indicator(df: pd.DataFrame, item_codes: list,
                      portfolio: int, mod: int = 0, risk: int = 0) -> pd.Series:
    """
    Filter rows matching any of the item codes and the given portfolio/
    modprod/risk combination, then return a Series indexed by LEI_Code
    with the summed Amount (multiple periods are averaged if present).
    """
    mask = (
        df["Item"].isin(item_codes) &
        (df["Portfolio"]    == portfolio) &
        (df["MKT_Modprod"]  == mod) &
        (df["Mkt_risk"]     == risk)
    )
    sub = df.loc[mask].groupby(["LEI_Code", "Period"])["Amount"].sum().reset_index()
    # If multiple periods: take the latest
    latest = sub.sort_values("Period").groupby("LEI_Code")["Amount"].last()
    return latest


def build_bank_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long data into one row per bank with key indicators as columns."""
    print("Pivoting to bank-level table ...")

    codes = {k: [c for c in v[1]] for k, v in ITEM_LABELS.items()}

    banks = pd.DataFrame(index=df["LEI_Code"].unique())
    banks.index.name = "LEI_Code"

    # Country
    banks["country"] = (df.drop_duplicates("LEI_Code")
                          .set_index("LEI_Code")["NSA"])

    # Total RWA — store both portfolios separately.
    # Portfolio=1: standardised approach (all banks)
    # Portfolio=5: IMA — correct denominator for VaR/SVaR ratios
    rwa_std = extract_indicator(df, codes["total_rwa"], portfolio=1)
    rwa_ima = extract_indicator(df, codes["total_rwa"], portfolio=5)
    banks["total_rwa_std"] = rwa_std
    banks["total_rwa_ima"] = rwa_ima
    # Use IMA RWA as primary total (more complete for IMA banks)
    banks["total_rwa"] = rwa_ima.combine_first(rwa_std)

    # VaR & SVaR — IMA banks only (Portfolio=5)
    for key in ["var_60d", "var_prev", "svar_60d", "svar_prev"]:
        banks[key] = extract_indicator(df, codes[key], portfolio=5)

    # IRC & CTP — Portfolio=5 (IMA capital charges, confirmed from EBA data)
    for key in ["irc_12w", "irc_last", "ctp_12w", "ctp_last"]:
        banks[key] = extract_indicator(df, codes[key], portfolio=5)

    banks = banks.reset_index()
    print(f"  Bank table: {len(banks)} banks, {banks.shape[1]} columns")
    return banks


# ---------------------------------------------------------------------------
# 4. DERIVED INDICATORS & TRAFFIC-LIGHT FLAGS
# ---------------------------------------------------------------------------

def engineer_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived ratios and apply traffic-light flags."""

    # VaR as % of IMA RWA (correct denominator: VaR is an IMA metric)
    df["var_to_rwa_pct"] = (df["var_60d"] / df["total_rwa_ima"].replace(0, np.nan)) * 100

    # SVaR / VaR stress multiplier
    df["svar_var_ratio"] = df["svar_60d"] / df["var_60d"].replace(0, np.nan)

    # IRC as % of total RWA
    df["irc_to_rwa_pct"] = (df["irc_12w"] / df["total_rwa"]) * 100

    # Replace inf with nan for flagging (zero RWA edge case → flagged separately)
    for col in ["var_to_rwa_pct", "svar_var_ratio", "irc_to_rwa_pct"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Traffic-light flags
    def flag(val, thr):
        if pd.isna(val): return "grey"
        if thr["higher_riskier"]:
            if val <= thr["warn"]:   return "green"
            if val <= thr["breach"]: return "amber"
            return "red"
        else:
            if val >= thr["warn"]:   return "green"
            if val >= thr["breach"]: return "amber"
            return "red"

    for ind, thr in THRESHOLDS.items():
        df[f"{ind}_flag"] = df[ind].apply(lambda v: flag(v, thr))

    # Overall risk score
    score_map = {"red": 2, "amber": 1, "green": 0, "grey": 0}
    flag_cols = [f"{i}_flag" for i in THRESHOLDS]
    df["risk_score"] = df[flag_cols].apply(
        lambda row: sum(score_map[v] for v in row), axis=1
    )
    df["risk_tier"] = pd.cut(
        df["risk_score"], bins=[-1, 0, 2, np.inf],
        labels=["Low", "Medium", "High"]
    )
    return df


# ---------------------------------------------------------------------------
# 5. VISUALISATION
# ---------------------------------------------------------------------------

PLOT_INDICATORS = {
    "var_to_rwa_pct":  "VaR (60d) as % of Market RWA",
    "svar_var_ratio":  "SVaR / VaR Stress Ratio",
    "irc_to_rwa_pct":  "IRC (12w) as % of Market RWA",
}

def plot_report(df: pd.DataFrame) -> None:
    """Multi-panel chart: polished distribution + flag breakdown per indicator."""
    import matplotlib.ticker as mticker

    n = len(PLOT_INDICATORS)
    fig = plt.figure(figsize=(18, 5 * n), facecolor=BG_COLOR)
    fig.suptitle(
        "EBA Transparency Exercise  ·  Market Risk Overview",
        fontsize=16, fontweight="bold", color=ACCENT, y=1.02,
        fontfamily="DejaVu Sans"
    )
    fig.text(0.5, 0.995, "IMA banks · latest available period",
             ha="center", fontsize=10, color="#6B7A99", fontfamily="DejaVu Sans")

    gs = GridSpec(n, 2, figure=fig, hspace=0.65, wspace=0.32,
                  left=0.07, right=0.96, top=0.96, bottom=0.04)

    flag_order  = ["green", "amber", "red", "grey"]
    flag_labels = {"green": "Within range", "amber": "Warning",
                   "red": "Breach", "grey": "No data"}

    for i, (ind, title) in enumerate(PLOT_INDICATORS.items()):
        flag_col = f"{ind}_flag"
        thr      = THRESHOLDS[ind]
        sub      = df.dropna(subset=[ind])
        n_inf    = (~np.isfinite(sub[ind])).sum()
        sub      = sub[np.isfinite(sub[ind])]

        # ── LEFT: stacked histogram ─────────────────────────────────────────
        ax_h = fig.add_subplot(gs[i, 0])
        ax_h.set_facecolor(PANEL_COLOR)
        ax_h.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax_h.set_axisbelow(True)

        for spine in ax_h.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        for flag in flag_order:
            vals = sub.loc[sub[flag_col] == flag, ind]
            if not vals.empty:
                ax_h.hist(vals, bins=18, color=COLORS[flag],
                          alpha=0.88, edgecolor="white", linewidth=0.5,
                          label=flag_labels[flag], zorder=3)

        # threshold lines
        ax_h.axvline(thr["warn"],   color=COLORS["amber"], ls="--",
                     lw=1.6, label=f"Warning ({thr['warn']})", zorder=4)
        ax_h.axvline(thr["breach"], color=COLORS["red"],   ls="--",
                     lw=1.6, label=f"Breach ({thr['breach']})",  zorder=4)

        inf_note = f"  (excl. {n_inf} with zero RWA)" if n_inf > 0 else ""
        ax_h.set_title(f"{title}{inf_note}",
                       fontsize=9.5, fontweight="bold", color=ACCENT, pad=8)
        ax_h.set_xlabel(title, fontsize=8, color="#6B7A99")
        ax_h.set_ylabel("Number of banks", fontsize=8, color="#6B7A99")
        ax_h.tick_params(colors="#6B7A99", labelsize=8)
        ax_h.legend(fontsize=7.5, framealpha=0.9, edgecolor=GRID_COLOR,
                    loc="upper right", ncol=2)

        # ── RIGHT: rounded bar chart ────────────────────────────────────────
        ax_b = fig.add_subplot(gs[i, 1])
        ax_b.set_facecolor(PANEL_COLOR)
        ax_b.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax_b.set_axisbelow(True)
        for spine in ax_b.spines.values():
            spine.set_edgecolor(GRID_COLOR)

        counts = (sub[flag_col].value_counts()
                    .reindex(flag_order, fill_value=0))
        x_pos  = np.arange(len(flag_order))
        bars   = ax_b.bar(
            x_pos, counts.values,
            color=[COLORS[f] for f in flag_order],
            width=0.55, edgecolor="white", linewidth=0.8, zorder=3
        )
        ax_b.bar_label(bars, fmt="%d", fontsize=9,
                       fontweight="bold", padding=4, color=ACCENT)
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels([flag_labels[f] for f in flag_order],
                              fontsize=8, color="#6B7A99")
        ax_b.set_title(f"{title} · Flag Breakdown",
                       fontsize=9.5, fontweight="bold", color=ACCENT, pad=8)
        ax_b.set_ylabel("Number of banks", fontsize=8, color="#6B7A99")
        ax_b.tick_params(colors="#6B7A99", labelsize=8)
        ax_b.set_ylim(0, counts.max() * 1.18 + 1)

    plt.savefig(OUTPUT_CHART, dpi=160, bbox_inches="tight",
                facecolor=BG_COLOR)
    print(f"  Chart saved → {OUTPUT_CHART}")
    plt.close()


# ---------------------------------------------------------------------------
# 6. EMAIL REPORT
# ---------------------------------------------------------------------------

import smtplib
import datetime as dt
from email.message import EmailMessage


def set_msip_label(msg, label_name):
    """Apply Microsoft Information Protection sensitivity label to email."""
    label_map = {
        'NON-BUSINESS':                      'ad9ed4cf-0835-41f5-8787-3905daafc7d0',
        'ECB-PUBLIC \\ Label':               '98056ac5-41a5-4345-a037-477d50cd8e4f',
        'ECB-PUBLIC \\ No Visible Label':    '21134413-d21c-4545-ac19-988c6c43347a',
        'ECB-UNRESTRICTED \\ Label':         'a9004f6e-0cb0-4bb2-8c11-22112a056e1d',
        'ECB-UNRESTRICTED \\ No Visible Label': 'df691197-c22c-4768-b7dc-9825d042eaaa',
        'ECB-RESTRICTED':                    '23da18b0-dae3-4c1e-8278-86f688a3028c',
        'ECB-CONFIDENTIAL \\ Business':      '894145f4-aec7-4462-867f-61720b9452af',
        'ECB-CONFIDENTIAL \\ Personal':      '8a9dd72c-31dc-456b-9fe8-03bdc3871139',
    }
    if label_name not in label_map:
        raise ValueError(f"Unknown label name '{label_name}'")
    label_guid = label_map[label_name]
    label_time = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    msip_header = (
        f'MSIP_Label_{label_guid}_ContentBits=0;'
        f'MSIP_Label_{label_guid}_Enabled=true;'
        f'MSIP_Label_{label_guid}_Method=Privileged;'
        f'MSIP_Label_{label_guid}_Name={label_name};'
        f'MSIP_Label_{label_guid}_SetDate={label_time};'
        f'MSIP_Label_{label_guid}_SiteId=b84ee435-4816-49d2-8d92-e740dbda4064;'
    )
    msg.add_header('msip_labels', msip_header)
    return msg


def send_html_email(to_address: str, html_message: str,
                    subject: str, output_file_path: str) -> None:
    """Send an HTML email via ECB SMTP and save a local copy for debugging.

    Args:
        to_address:       Recipient email address.
        html_message:     HTML string with the email body.
        subject:          Email subject line.
        output_file_path: Path where a local .html copy is saved.
    """
    # Save local copy for debugging / audit trail
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(html_message)

    msg = EmailMessage()
    msg = set_msip_label(msg, 'ECB-CONFIDENTIAL \\ Business')
    msg["From"]    = "paola.bignoli@ecb.europa.eu"
    msg["To"]      = to_address
    msg["Subject"] = subject

    msg.set_content("This email requires an HTML-compatible client to view.")
    msg.add_alternative(html_message, subtype="html")

    with smtplib.SMTP("mail-exception0-ip-gw.ecb.de") as s:
        s.send_message(msg)
    print(f"  Email sent → {to_address}")


def build_html_report(summary_df: pd.DataFrame) -> str:
    """Build the HTML body for the market risk email report."""
    n_banks   = len(summary_df)
    n_ima     = summary_df["var_60d"].notna().sum()
    high_risk = (summary_df["risk_tier"] == "High").sum()

    def flag_count(col, flag):
        return (summary_df.get(col, pd.Series(dtype=str)) == flag).sum()

    var_green  = flag_count("var_to_rwa_pct_flag", "green")
    var_amber  = flag_count("var_to_rwa_pct_flag", "amber")
    var_red    = flag_count("var_to_rwa_pct_flag", "red")
    svar_green = flag_count("svar_var_ratio_flag", "green")
    svar_amber = flag_count("svar_var_ratio_flag", "amber")
    svar_red   = flag_count("svar_var_ratio_flag", "red")
    irc_green  = flag_count("irc_to_rwa_pct_flag", "green")
    irc_amber  = flag_count("irc_to_rwa_pct_flag", "amber")
    irc_red    = flag_count("irc_to_rwa_pct_flag", "red")

    kpi_boxes = "".join([
        f'''<td style="width:25%; padding:0 8px;">
          <div style="background:{bg}; border-radius:8px; padding:16px;
                      text-align:center; border-left:4px solid {bc};">
            <div style="font-size:26px; font-weight:bold; color:{fc};">{val}</div>
            <div style="font-size:11px; color:#666; margin-top:4px;">{lbl}</div>
          </div></td>'''
        for bg, bc, fc, val, lbl in [
            ("#EAF4EE", "#4CAF8A", "#2E7D52", n_ima,     "IMA Banks"),
            ("#FFF8EC", "#F5A623", "#B96A00", var_amber,  "VaR – Warning"),
            ("#FDECEA", "#E05C5C", "#C0392B", var_red,    "VaR – Breach"),
            ("#EEF2FB", "#2C3E6B", "#2C3E6B", high_risk,  "High Risk Tier"),
        ]
    ])

    table_rows = "".join([
        f'''<tr style="border-top:1px solid #EEE;">
          <td style="padding:9px 14px; color:#444;">{name}</td>
          <td style="text-align:center; padding:9px; color:#2E7D52; font-weight:bold;">{g}</td>
          <td style="text-align:center; padding:9px; color:#B96A00; font-weight:bold;">{a}</td>
          <td style="text-align:center; padding:9px; color:#C0392B; font-weight:bold;">{r}</td>
        </tr>'''
        for name, g, a, r in [
            ("VaR (60d) / IMA RWA",  var_green,  var_amber,  var_red),
            ("SVaR / VaR Ratio",     svar_green, svar_amber, svar_red),
            ("IRC (12w) / IMA RWA",  irc_green,  irc_amber,  irc_red),
        ]
    ])

    return f"""<!DOCTYPE html>
<html><body style="margin:0; padding:30px; background:#F0F4F8;
                   font-family:Arial, sans-serif;">
  <div style="max-width:680px; margin:auto; background:#fff;
              border-radius:10px; overflow:hidden;
              box-shadow:0 2px 14px rgba(0,0,0,0.09);">

    <!-- Header -->
    <div style="background:#2C3E6B; padding:28px 32px;">
      <h1 style="color:#fff; margin:0; font-size:20px; letter-spacing:0.3px;">
        EBA Transparency Exercise</h1>
      <p style="color:#A8C0E8; margin:6px 0 0; font-size:13px;">
        Market Risk Overview &nbsp;·&nbsp; Automated Report
        &nbsp;·&nbsp; {dt.datetime.now().strftime("%d %b %Y")}</p>
    </div>

    <!-- Intro -->
    <div style="padding:26px 32px 10px;">
      <p style="color:#444; font-size:14px; line-height:1.75; margin:0;">
        This report summarises market risk indicators for
        <strong>{n_banks} European banks</strong> using the latest available
        EBA Transparency Exercise data (<em>tr_mrk.csv</em>). Indicators cover
        Value-at-Risk, Stressed VaR, and Incremental Risk Charge, each
        normalised to the bank's Internal Models Approach (IMA) Risk-Weighted
        Assets. Traffic-light flags are applied based on supervisory thresholds
        to support early identification of elevated market risk profiles.
      </p>
    </div>

    <!-- KPI row -->
    <div style="padding:20px 32px;">
      <table style="width:100%; border-collapse:separate; border-spacing:0;">
        <tr>{kpi_boxes}</tr>
      </table>
    </div>

    <!-- Indicator table -->
    <div style="padding:0 32px 26px;">
      <h3 style="color:#2C3E6B; font-size:14px; margin:0 0 10px;">
        Indicator Flag Summary</h3>
      <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
          <tr style="background:#F0F4F8;">
            <th style="text-align:left; padding:9px 14px; color:#2C3E6B;">Indicator</th>
            <th style="padding:9px; color:#2E7D52;">&#9679; Within range</th>
            <th style="padding:9px; color:#B96A00;">&#9679; Warning</th>
            <th style="padding:9px; color:#C0392B;">&#9679; Breach</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>

    <!-- Chart note -->
    <div style="padding:0 32px 26px;">
      <p style="color:#666; font-size:13px; line-height:1.6; margin:0;">
        The full distribution chart is attached to this email as a PNG file.
        It shows, for each indicator, the cross-bank distribution and the
        flag breakdown across all IMA institutions in the sample.
      </p>
    </div>

    <!-- Footer -->
    <div style="background:#F0F4F8; padding:14px 32px;
                border-top:1px solid #E8ECF0; text-align:center;">
      <p style="color:#aaa; font-size:11px; margin:0;">
        Generated automatically &nbsp;·&nbsp;
        EBA Market Risk Module &nbsp;·&nbsp;
        For internal use only
      </p>
    </div>

  </div>
</body></html>"""


def send_email_report(chart_path: str, summary_df: pd.DataFrame) -> None:
    """Build and send the HTML market risk report email with chart attachment."""
    receiver       = "bignolipaola@gmail.com"
    subject        = "EBA Market Risk Report – Automated Overview"
    local_copy     = "eba_market_risk_email.html"
    html           = build_html_report(summary_df)

    # Add chart as attachment by appending to the EmailMessage after send_html_email
    with open("eba_market_risk_email.html", "w", encoding="utf-8") as f:
        f.write(html)

    try:
        msg = EmailMessage()
        msg = set_msip_label(msg, "ECB-CONFIDENTIAL \\ Business")
        msg["From"]    = "paola.bignoli@ecb.europa.eu"
        msg["To"]      = receiver
        msg["Subject"] = subject
        msg.set_content("This email requires an HTML-compatible client to view.")
        msg.add_alternative(html, subtype="html")

        # Attach chart as PNG
        with open(chart_path, "rb") as f:
            chart_data = f.read()
        msg.add_attachment(chart_data, maintype="image", subtype="png",
                           filename="eba_market_risk_chart.png")

        with smtplib.SMTP("mail-exception0-ip-gw.ecb.de") as s:
            s.send_message(msg)
        print(f"  Email sent → {receiver}")
    except Exception as e:
        print(f"  Email failed: {e}")
        print(f"  HTML saved locally → {local_copy}")
# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n=== EBA Market Risk Analyser ===\n")

    df_raw   = load_data(INPUT_FILE)
    df_banks = build_bank_table(df_raw)
    df_banks = engineer_and_flag(df_banks)

    # Export ranked summary
    out_cols = (
        ["LEI_Code", "country", "total_rwa",
         "var_60d", "var_prev", "svar_60d", "svar_prev",
         "irc_12w", "ctp_12w",
         "var_to_rwa_pct", "svar_var_ratio", "irc_to_rwa_pct"] +
        [f"{i}_flag" for i in THRESHOLDS] +
        ["risk_score", "risk_tier"]
    )
    out_cols = [c for c in out_cols if c in df_banks.columns]
    (df_banks[out_cols]
        .sort_values("risk_score", ascending=False)
        .to_csv(OUTPUT_CSV, index=False))
    print(f"  Summary CSV saved → {OUTPUT_CSV}")

    # High-risk banks
    high = df_banks[df_banks["risk_tier"] == "High"]
    print(f"\n  High-risk banks (market risk): {len(high)}")
    if not high.empty and "country" in high.columns:
        print(high[["LEI_Code", "country", "risk_score",
                     "var_to_rwa_pct", "svar_var_ratio"]].to_string(index=False))

    plot_report(df_banks)

    # Send email report (requires EMAIL_SENDER and EMAIL_PASSWORD env vars)
    send_email_report(OUTPUT_CHART, df_banks)
    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
