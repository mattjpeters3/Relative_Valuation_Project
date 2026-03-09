import os
import time
import logging
import pandas as pd
import yfinance as yf
from config.paths import INDIVIDUAL_SP500_STOCK_DATA_FOLDER, ALL_SP500_STOCK_DATA_FOLDER
from config.tickers import get_sp500_tickers, get_sp500_tickers_and_sectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("stock_data.log"),
        logging.StreamHandler()
    ]
)

def download_stock_metrics(original_ticker):
    try:
        yf_ticker = original_ticker.replace('.', '-')
        stock = yf.Ticker(yf_ticker)
        info = stock.info

        # ── EPS and PE ────────────────────────────────────────────────────
        prev_close = info.get('previousClose')
        eps        = info.get('epsTrailingTwelveMonths')

        # PE is only meaningful when earnings are positive.
        # Negative or zero EPS produces a meaningless or negative PE multiple
        # so we record None and the cleaning step will drop these firms.
        if prev_close and eps and eps > 0:
            pe_ratio_current = prev_close / eps
        else:
            pe_ratio_current = None

        # Drop firms where PE > 100. At these levels earnings are so thin
        # that PE comparison is not meaningful — the multiple is dominated
        # by near-zero earnings rather than real valuation differences.
        if pe_ratio_current is not None and pe_ratio_current > 100:
            pe_ratio_current = None

        # ── Payout Ratio and Dividend Yield ───────────────────────────────
        payout_ratio   = info.get('payoutRatio')
        dividend_yield = info.get('dividendYield')

        # Distinguish genuine non-payers from firms with no earnings:
        #   - EPS > 0 and payout_ratio is None → genuine non-payer → treat as 0
        #   - EPS <= 0 and payout_ratio is None → undefined ratio   → keep None
        #     (will be dropped in cleaning since PE is also None)
        if payout_ratio is None and eps is not None and eps > 0:
            payout_ratio = 0.0

        # Dividend yield: None always means no dividend → treat as 0
        # (used in clustering only, not in regression)
        if dividend_yield is None:
            dividend_yield = 0.0

        # ── Retention Ratio and EPS Growth ───────────────────────────────
        roe = info.get('returnOnEquity')

        retention_ratio = 1 - payout_ratio if payout_ratio is not None else None

        # EPS growth via Gordon Growth sustainable growth formula: ROE × Retention
        # Keep negative values (shrinking earnings base is real information).
        # Drop only if ROE is missing — EPS growth is a core regression predictor.
        # Drop if result is outside [-1.0, 1.0]: beyond these bounds the Gordon
        # Growth formula has broken down (data quality issue or financial distress).
        if retention_ratio is not None and roe is not None:
            raw_growth = retention_ratio * roe
            if -1.0 <= raw_growth <= 1.0:
                eps_growth = raw_growth
            else:
                eps_growth = None  # outside plausible range — exclude
                logging.warning(
                    f"{original_ticker}: EPS growth ({raw_growth:.2f}) outside [-1, 1] — excluded"
                )
        else:
            eps_growth = None

        metrics = {
            'Ticker':                        original_ticker,
            'Market Cap':                    info.get('marketCap'),
            'PE Ratio (TTM)':                info.get('trailingPE'),
            'PE Ratio (Current)':            pe_ratio_current,
            'Dividend Yield':                dividend_yield,
            'Beta':                          info.get('beta'),
            'Payout Ratio':                  payout_ratio,
            'Return on Equity (ROE)':        roe,
            'Retention Ratio':               retention_ratio,
            'EPS Growth (ROE x Retention)':  eps_growth,
        }

        return pd.DataFrame([metrics])

    except Exception as e:
        logging.error(f"Failed to fetch data for {original_ticker}: {e}")
        return None

def save_csv(df, path):
    try:
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path}")
    except Exception as e:
        logging.error(f"Failed to save CSV to {path}: {e}")


def clean_and_save_filtered_data(df):
    try:
        total = len(df)
        report_lines = [
            "",
            "=" * 60,
            "  DATA CLEANING REPORT",
            "=" * 60,
            f"  Firms fetched from Yahoo Finance : {total}",
        ]

        # ── Step 1: Drop firms where PE is missing or was set to None ────
        # This covers: negative EPS, zero EPS, PE > 100, fetch failures.
        step1 = df.dropna(subset=['PE Ratio (Current)'])
        dropped_pe = total - len(step1)
        report_lines.append(
            f"  Dropped — PE missing/invalid     : {dropped_pe}"
            f"  (negative EPS, near-zero EPS, PE > 100, or fetch failure)"
        )

        # ── Step 2: Drop firms where EPS Growth is missing ───────────────
        # Covers: missing ROE, EPS growth outside [-1, 1] (Gordon breakdown)
        step2 = step1.dropna(subset=['EPS Growth (ROE x Retention)'])
        dropped_eps = len(step1) - len(step2)
        report_lines.append(
            f"  Dropped — EPS growth missing     : {dropped_eps}"
            f"  (missing ROE or growth outside plausible range)"
        )

        # ── Step 3: Drop firms where Beta is missing ─────────────────────
        step3 = step2.dropna(subset=['Beta'])
        dropped_beta = len(step2) - len(step3)
        report_lines.append(
            f"  Dropped — Beta missing           : {dropped_beta}"
        )

        # ── Step 4: Drop firms where Payout Ratio is missing ─────────────
        # After our imputation in download_stock_metrics, this only catches
        # firms with negative EPS where we intentionally left payout as None.
        step4 = step3.dropna(subset=['Payout Ratio'])
        dropped_payout = len(step3) - len(step4)
        report_lines.append(
            f"  Dropped — Payout ratio missing   : {dropped_payout}"
            f"  (negative EPS — PE valuation not applicable)"
        )

        # ── Step 5: Drop negative Retention Ratio ────────────────────────
        # Payout ratio > 1 means firm pays out more than it earns
        step5 = step4[step4['Retention Ratio'] >= 0]
        dropped_retention = len(step4) - len(step5)
        report_lines.append(
            f"  Dropped — Retention ratio < 0    : {dropped_retention}"
            f"  (payout ratio > 100% — unsustainable dividend)"
        )

        # ── Step 6: Drop remaining missing values ─────────────────────────
        # Market Cap, Beta, Standard Deviation, ROE
        core_cols = [c for c in step5.columns if c not in ('Sector',)]
        step6 = step5.dropna(subset=core_cols)
        dropped_other = len(step5) - len(step6)
        report_lines.append(
            f"  Dropped — other missing fields   : {dropped_other}"
            f"  (Market Cap, ROE, etc.)"
        )

        cleaned_df = step6.copy()

        # Fill any missing Sector with 'Unknown'
        if 'Sector' in cleaned_df.columns:
            cleaned_df['Sector'] = cleaned_df['Sector'].fillna('Unknown')

        report_lines += [
            f"  {'─' * 46}",
            f"  Firms retained for modelling     : {len(cleaned_df)} of {total}",
            f"  Attrition rate                   : {(total - len(cleaned_df)) / total * 100:.1f}%",
            "=" * 60,
            "",
        ]

        for line in report_lines:
            logging.info(line)

        save_csv(cleaned_df, os.path.join(ALL_SP500_STOCK_DATA_FOLDER, "cleaned_combined_sp500_data.csv"))

    except Exception as e:
        logging.error(f"Failed to clean and save data: {e}")


def main():
    # Fetch tickers with sector labels from Wikipedia
    ticker_sector_df = get_sp500_tickers_and_sectors()
    sp500_tickers    = ticker_sector_df['Ticker'].tolist()

    if not sp500_tickers:
        logging.error("No tickers found, exiting.")
        return

    combined_data = []

    for i, ticker in enumerate(sp500_tickers):
        logging.info(f"Fetching data for {ticker} ({i + 1}/{len(sp500_tickers)})...")
        df = download_stock_metrics(ticker)
        if df is not None:
            save_csv(df, os.path.join(INDIVIDUAL_SP500_STOCK_DATA_FOLDER, f"{ticker}.csv"))
            combined_data.append(df)
        time.sleep(1)

    if combined_data:
        all_data = pd.concat(combined_data, ignore_index=True)

        # Merge sector labels onto combined data
        all_data = all_data.merge(ticker_sector_df, on='Ticker', how='left')

        save_csv(all_data, os.path.join(ALL_SP500_STOCK_DATA_FOLDER, "combined_sp500_data.csv"))
        clean_and_save_filtered_data(all_data)
    else:
        logging.warning("No data downloaded; combined file not created.")

if __name__ == "__main__":
    main()