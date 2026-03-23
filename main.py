import os
import sys

from config.paths import (
    INDIVIDUAL_SP500_STOCK_DATA_FOLDER,
    ALL_SP500_STOCK_DATA_FOLDER,
    STOCK_CLUSTERS_FOLDER,
    PREDICTED_PE_RATIO_RESULTS,
)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def clear_folder(folder_path: str, label: str, preserve: list = None) -> None:
    """
    Delete all files inside folder_path without removing the folder itself.
    Subdirectories are left untouched.

    preserve: optional list of filenames to skip (e.g. ['signal_history.csv'])
    """
    if not os.path.exists(folder_path):
        print(f"  [skip] {label} folder does not exist yet.")
        return

    preserve = preserve or []
    removed = 0
    for filename in os.listdir(folder_path):
        if filename in preserve:
            continue
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed += 1

    print(f"  Cleared {removed} file(s) from {label}.")


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# PIPELINE STAGES
# ---------------------------------------------------------------------------

def run_data_collection() -> None:
    """
    Stage 1 — Download fresh financial data for all S&P 500 constituents.

    Clears stale individual stock files and the combined/cleaned CSVs
    before fetching, so the dataset always reflects the current index.
    """
    section("STAGE 1: Data Collection")

    print("\nClearing stale data files...")
    clear_folder(INDIVIDUAL_SP500_STOCK_DATA_FOLDER, "individual stock data")
    clear_folder(ALL_SP500_STOCK_DATA_FOLDER,        "combined stock data")

    from download_data.stock_data import main as fetch_data
    fetch_data()


def run_clustering() -> None:
    """
    Stage 2 — Cluster S&P 500 firms by fundamental characteristics.

    Clears old cluster CSVs first so stale clusters from a previous
    K value cannot contaminate the regression downstream.
    """
    section("STAGE 2: Clustering")

    print("\nClearing stale cluster files...")
    clear_folder(STOCK_CLUSTERS_FOLDER, "stock clusters")

    from comparable_firms.stock_clustering import main as cluster
    cluster()


def run_valuation() -> None:
    """
    Stage 3 — Run cluster and whole-index regressions, generate
    predicted PE ratios, assign valuation signals, and produce the
    master valuation table.

    Clears old prediction files first to prevent stale cluster results
    from being merged into the new master table.
    """
    section("STAGE 3: Valuation")

    print("\nClearing stale prediction files...")
    clear_folder(
        PREDICTED_PE_RATIO_RESULTS,
        "predicted PE results",
        preserve=["signal_history.csv", "paper_positions.csv", "paper_positions_cluster.csv"],
    )

    from predicted_ratios.predicted_pe_ratio import (
        calculate_predicted_pe_ratios,
        calculate_whole_index_pe_ratios,
        combine_and_filter_results,
    )

    print("\nStep 1: Running per-cluster regressions...")
    calculate_predicted_pe_ratios()

    print("\nStep 2: Running whole-index regression...")
    calculate_whole_index_pe_ratios()

    print("\nStep 3: Merging signals into master valuation table...")
    combine_and_filter_results()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

STAGES = {
    "1": ("Data Collection",  run_data_collection),
    "2": ("Clustering",       run_clustering),
    "3": ("Valuation",        run_valuation),
}


def print_menu() -> None:
    print("\n" + "=" * 60)
    print("  RELATIVE VALUATION PIPELINE")
    print("=" * 60)
    print("  Select which stages to run:\n")
    print("  [1] Data Collection   — fetch fresh S&P 500 data")
    print("  [2] Clustering        — group firms by fundamentals")
    print("  [3] Valuation         — run regressions & generate signals")
    print("  [all] Run full pipeline (1 → 2 → 3)")
    print("  [q]   Quit")
    print("=" * 60)


if __name__ == "__main__":

    # If a command-line argument is passed (e.g. `python main.py all`),
    # use it directly without showing the menu.
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
    else:
        print_menu()
        choice = input("\nEnter choice: ").strip().lower()

    if choice == "q":
        print("Exiting.")
        sys.exit(0)

    elif choice == "all":
        for key in ["1", "2", "3"]:
            label, fn = STAGES[key]
            fn()
        print("\nFull pipeline complete.")

    elif choice in STAGES:
        label, fn = STAGES[choice]
        fn()
        print(f"\n{label} complete.")

    else:
        print(f"Unknown choice '{choice}'. Valid options: 1, 2, 3, all, q")
        sys.exit(1)

"""
git add . 
git commit -m "Refresh data " 
git push

"""
