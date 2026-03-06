import io
import requests
import pandas as pd


def get_sp500_tickers() -> list[str]:
    """
    Fetch the current S&P 500 constituent tickers dynamically from Wikipedia.

    Wikipedia's S&P 500 page maintains an up-to-date table of all index
    members and is updated within days of any composition change.

    Uses requests with a browser User-Agent header to avoid Wikipedia's
    bot-blocking, and passes the raw HTML to pd.read_html() to bypass
    the macOS Python 3.13 SSL certificate issue.

    Returns
    -------
    list[str]
        Ticker symbols for all current S&P 500 constituents, with dots
        replaced by hyphens to match Yahoo Finance's format
        (e.g. 'BRK.B' -> 'BRK-B').

    Raises
    ------
    RuntimeError
        If the Wikipedia page cannot be reached or the expected table
        structure is not found.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not fetch S&P 500 constituents from Wikipedia: {e}\n"
            "Check your internet connection and try again."
        )

    try:
        tables = pd.read_html(io.StringIO(response.text))
    except Exception as e:
        raise RuntimeError(f"Could not parse Wikipedia HTML table: {e}")

    constituents = tables[0]

    if 'Symbol' not in constituents.columns:
        raise RuntimeError(
            "Wikipedia table structure has changed — 'Symbol' column not found. "
            "Check the page manually and update the column name if needed."
        )

    tickers = (
        constituents['Symbol']
        .str.strip()
        .str.replace('.', '-', regex=False)
        .tolist()
    )

    return tickers


def get_sp500_tickers_and_sectors() -> pd.DataFrame:
    """
    Fetch current S&P 500 tickers AND their GICS sectors from Wikipedia.

    Returns
    -------
    pd.DataFrame
        Columns: 'Ticker', 'Sector'
        One row per S&P 500 constituent.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not fetch S&P 500 constituents from Wikipedia: {e}"
        )

    try:
        tables = pd.read_html(io.StringIO(response.text))
    except Exception as e:
        raise RuntimeError(f"Could not parse Wikipedia HTML table: {e}")

    constituents = tables[0]

    if 'Symbol' not in constituents.columns:
        raise RuntimeError("Wikipedia table structure has changed — 'Symbol' column not found.")

    # GICS Sector column name may vary slightly
    sector_col = next(
        (c for c in constituents.columns if 'sector' in c.lower()),
        None
    )

    df = pd.DataFrame()
    df['Ticker'] = (
        constituents['Symbol']
        .str.strip()
        .str.replace('.', '-', regex=False)
    )
    df['Sector'] = constituents[sector_col].str.strip() if sector_col else 'Unknown'

    return df.reset_index(drop=True)


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Successfully fetched {len(tickers)} S&P 500 tickers.")
    print(f"First 10: {tickers[:10]}")
    print(f"Last 10:  {tickers[-10:]}")

    df = get_sp500_tickers_and_sectors()
    print(f"\nSector breakdown:")
    print(df['Sector'].value_counts().to_string())