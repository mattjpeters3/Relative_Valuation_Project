import os

# Base directory: folder where this file (paths.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root: one level up from config folder (Relative_Valuation)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Define core folders
CONFIG_FOLDER = os.path.join(PROJECT_ROOT, "config")
DOWNLOAD_DATA_FOLDER = os.path.join(PROJECT_ROOT, "download_data")
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
COMPARABLE_FIRMS_FOLDER = os.path.join(PROJECT_ROOT, "comparable_firms")
PREDICTED_RATIOS_FOLDER = os.path.join(PROJECT_ROOT, "predicted_ratios")
BACKTESTING_FOLDER = os.path.join(PROJECT_ROOT, "backtesting")
PAPER_TRADING_FOLDER = os.path.join(PROJECT_ROOT, "paper_trading")

# Subfolders inside DATA_FOLDER
LARGE_CAP_STOCK_DATA_FOLDER = os.path.join(DATA_FOLDER, "large_cap_stock_data")
MID_CAP_STOCK_DATA_FOLDER = os.path.join(DATA_FOLDER, "mid_cap_stock_data")
SMALL_CAP_STOCK_DATA_FOLDER = os.path.join(DATA_FOLDER, "small_cap_stock_data")
SP500_STOCK_DATA_FOLDER = os.path.join(DATA_FOLDER, "s&p500_stock_data")

# Subfolders inside LARGE_CAP_STOCK_DATA_FOLDER
INDIVIDUAL_LARGE_CAP_STOCK_DATA_FOLDER = os.path.join(LARGE_CAP_STOCK_DATA_FOLDER, "individual_large_cap_stock_data")
ALL_LARGE_CAP_STOCK_DATA_FOLDER = os.path.join(LARGE_CAP_STOCK_DATA_FOLDER, "all_large_cap_stock_data")

# Subfolders inside MID_CAP_STOCK_DATA_FOLDER
INDIVIDUAL_MID_CAP_STOCK_DATA_FOLDER = os.path.join(MID_CAP_STOCK_DATA_FOLDER, "individual_mid_cap_stock_data")
ALL_MID_CAP_STOCK_DATA_FOLDER = os.path.join(MID_CAP_STOCK_DATA_FOLDER, "all_mid_cap_stock_data")

# Subfolders inside SMALL_CAP_STOCK_DATA_FOLDER
INDIVIDUAL_SMALL_CAP_STOCK_DATA_FOLDER = os.path.join(SMALL_CAP_STOCK_DATA_FOLDER, "individual_small_cap_stock_data")
ALL_SMALL_CAP_STOCK_DATA_FOLDER = os.path.join(SMALL_CAP_STOCK_DATA_FOLDER, "all_small_cap_stock_data")

# Subfolders inside SP500_STOCK_DATA_FOLDER
INDIVIDUAL_SP500_STOCK_DATA_FOLDER = os.path.join(SP500_STOCK_DATA_FOLDER, "individual_SP500_stock_data")
ALL_SP500_STOCK_DATA_FOLDER = os.path.join(SP500_STOCK_DATA_FOLDER, "all_SP500_stock_data")

# Subfolders inside COMPARABLE_FIRMS_FOLDER
STOCK_CLUSTERS_FOLDER = os.path.join(COMPARABLE_FIRMS_FOLDER, "stock_clusters")

# Subfolders inside PREDICTED_RATIOS_FOLDER
PREDICTED_PE_RATIO_RESULTS = os.path.join(PREDICTED_RATIOS_FOLDER,"predicted_pe_ratio_results")

# Subfolders inside BACKTESTING_FOLDER
HISTORICAL_SNAPSHOTS_FOLDER = os.path.join(BACKTESTING_FOLDER,"historical_snapshots")
VALUATION_OUTPUTS_FOLDER = os.path.join(BACKTESTING_FOLDER, "valuation_outputs")
DAILY_PRICES_FOLDER = os.path.join(BACKTESTING_FOLDER, "daily_prices")
HISTORICAL_SP500_MEMBERSHIP_FOLDER = os.path.join(BACKTESTING_FOLDER, "historical_SP500_membership")
HISTORICAL_SP500_MEMBERSHIP_CSV = os.path.join(HISTORICAL_SP500_MEMBERSHIP_FOLDER, "sp500_membership.csv")
BACKTEST_PREDICTED_RATIOS = os.path.join(BACKTESTING_FOLDER, "backtest_predicted_ratios")

# Subfolders inside PAPER_TRADING_FOLDER
PORTFOLIO_FILE = os.path.join(PAPER_TRADING_FOLDER, "portfolio.csv")
TRADE_LOG_FILE = os.path.join(PAPER_TRADING_FOLDER, "trade_log.csv")


# Ensure all necessary folders exist
for path in [
    CONFIG_FOLDER,
    DOWNLOAD_DATA_FOLDER,
    DATA_FOLDER,
    LARGE_CAP_STOCK_DATA_FOLDER,
    MID_CAP_STOCK_DATA_FOLDER,
    SMALL_CAP_STOCK_DATA_FOLDER,
    SP500_STOCK_DATA_FOLDER,
    INDIVIDUAL_LARGE_CAP_STOCK_DATA_FOLDER,
    ALL_LARGE_CAP_STOCK_DATA_FOLDER,
    INDIVIDUAL_MID_CAP_STOCK_DATA_FOLDER,
    ALL_MID_CAP_STOCK_DATA_FOLDER,
    INDIVIDUAL_SMALL_CAP_STOCK_DATA_FOLDER,
    ALL_SMALL_CAP_STOCK_DATA_FOLDER,
    INDIVIDUAL_SP500_STOCK_DATA_FOLDER,
    ALL_SP500_STOCK_DATA_FOLDER,
    COMPARABLE_FIRMS_FOLDER,
    STOCK_CLUSTERS_FOLDER,
    PREDICTED_RATIOS_FOLDER,
    PREDICTED_PE_RATIO_RESULTS,
    BACKTESTING_FOLDER,
    HISTORICAL_SNAPSHOTS_FOLDER,
    VALUATION_OUTPUTS_FOLDER,
    DAILY_PRICES_FOLDER,
    HISTORICAL_SP500_MEMBERSHIP_FOLDER,
    BACKTEST_PREDICTED_RATIOS,
]:
    os.makedirs(path, exist_ok=True)

# Centralized dictionary for flexible access
PATHS = {
    "config": CONFIG_FOLDER,
    "download_data": DOWNLOAD_DATA_FOLDER,
    "data": DATA_FOLDER,
    "large_cap_stock_data": LARGE_CAP_STOCK_DATA_FOLDER,
    "mid_cap_stock_data": MID_CAP_STOCK_DATA_FOLDER,
    "small_cap_stock_data": SMALL_CAP_STOCK_DATA_FOLDER,
    "s&p500_stock_data": SP500_STOCK_DATA_FOLDER,
    "individual_large_cap_stock_data": INDIVIDUAL_LARGE_CAP_STOCK_DATA_FOLDER,
    "all_large_cap_stock_data": ALL_LARGE_CAP_STOCK_DATA_FOLDER,
    "individual_mid_cap_stock_data": INDIVIDUAL_MID_CAP_STOCK_DATA_FOLDER,
    "all_mid_cap_stock_data": ALL_MID_CAP_STOCK_DATA_FOLDER,
    "individual_small_cap_stock_data": INDIVIDUAL_SMALL_CAP_STOCK_DATA_FOLDER,
    "all_small_cap_stock_data": ALL_SMALL_CAP_STOCK_DATA_FOLDER,
    "individual_SP500_stock_data": INDIVIDUAL_SP500_STOCK_DATA_FOLDER,
    "all_SP500_stock_data": ALL_SP500_STOCK_DATA_FOLDER,
    "comparable_firms": COMPARABLE_FIRMS_FOLDER,
    "stock_clusters": STOCK_CLUSTERS_FOLDER,
    "predicted_ratios": PREDICTED_RATIOS_FOLDER,
    "predicted_pe_ratio_results": PREDICTED_PE_RATIO_RESULTS,
    "backtesting": BACKTESTING_FOLDER,
    "historical_snapshots": HISTORICAL_SNAPSHOTS_FOLDER,
    "valuation_outputs": VALUATION_OUTPUTS_FOLDER,
    "daily_prices": DAILY_PRICES_FOLDER,
    "sp500_membership.csv": HISTORICAL_SP500_MEMBERSHIP_CSV,
    "historical_SP500_membership": HISTORICAL_SP500_MEMBERSHIP_FOLDER,
    "backtest_predicted_ratios": BACKTEST_PREDICTED_RATIOS,


}