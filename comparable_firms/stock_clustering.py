import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from config.paths import PATHS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_CSV  = os.path.join(PATHS["all_SP500_stock_data"], "cleaned_combined_sp500_data.csv")
N_CLUSTERS = None   # Set to an integer to override; leave None to use elbow method

# Range of K values the elbow method will evaluate
ELBOW_K_MIN = 2
ELBOW_K_MAX = 20

# Features used for clustering.
# These represent the three fundamental value drivers:
#   EPS Growth  → expected cash flow growth
#   Beta        → systematic risk
#   Dividend Yield → cash flow distribution (proxy for payout policy)
CLUSTER_FEATURES = ['EPS Growth (ROE x Retention)', 'Beta', 'Dividend Yield']

# Columns saved to each per-cluster CSV (passed downstream to the regression)
COLUMNS_TO_SAVE = [
    'Ticker', 'PE Ratio (TTM)', 'PE Ratio (Current)',
    'Market Cap (log)', 'Beta', 'Dividend Yield',
    'EPS Growth (ROE x Retention)', 'Cluster', 'Sector',
]


# ---------------------------------------------------------------------------
# STEP 1: Load and prepare data
# ---------------------------------------------------------------------------

def load_and_prepare(input_csv: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load cleaned S&P 500 data, drop rows missing cluster features,
    add log market cap, and return the DataFrame plus the scaled feature matrix."""
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=CLUSTER_FEATURES).copy()
    df['Market Cap (log)'] = np.log(df['Market Cap'])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

    print(f"Loaded {len(df)} firms with complete clustering features.")
    return df, X_scaled


# ---------------------------------------------------------------------------
# STEP 2: Elbow method — find optimal K
# ---------------------------------------------------------------------------

def run_elbow_method(X_scaled: np.ndarray, k_min: int, k_max: int) -> int:
    """
    Fit KMeans for each K in [k_min, k_max], record within-cluster sum of
    squares (WCSS), plot the elbow curve, and return the elbow K.

    The elbow is identified using the 'kneedle' heuristic: the point of
    maximum curvature, found by measuring the perpendicular distance from
    each WCSS point to the straight line connecting the first and last points.
    This is a simple and robust automated elbow detection method.
    """
    k_values = list(range(k_min, k_max + 1))
    wcss     = []

    print(f"\nRunning elbow method for K = {k_min} to {k_max}...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        print(f"  K={k:>2}  WCSS={km.inertia_:>10.2f}")

    # --- Automated elbow detection (kneedle heuristic) ----------------------
    # Normalise both axes to [0, 1] so the geometry is scale-invariant
    x = np.array(k_values, dtype=float)
    y = np.array(wcss,     dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Vector from first to last point
    vec      = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    vec_norm = vec / np.linalg.norm(vec)

    # Perpendicular distance from each point to that line
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        pt  = np.array([xi - x_norm[0], yi - y_norm[0]])
        d   = abs(pt[0] * vec_norm[1] - pt[1] * vec_norm[0])
        distances.append(d)

    elbow_idx = int(np.argmax(distances))
    elbow_k   = k_values[elbow_idx]

    # --- Plot ---------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, wcss, marker='o', linewidth=2, color='steelblue', label='WCSS')
    plt.axvline(
        x=elbow_k, color='crimson', linestyle='--', linewidth=1.5,
        label=f'Elbow at K={elbow_k}'
    )
    plt.title('Elbow Method — Optimal Number of Clusters', fontsize=14)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    elbow_plot_path = os.path.join(PATHS["stock_clusters"], "elbow_plot.png")
    plt.savefig(elbow_plot_path, dpi=150)
    plt.close()
    print(f"\nElbow plot saved to: {elbow_plot_path}")
    print(f"Elbow method suggests K = {elbow_k}")

    return elbow_k


# ---------------------------------------------------------------------------
# STEP 3: Fit KMeans with chosen K
# ---------------------------------------------------------------------------

def fit_kmeans(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print(f"\nFitted KMeans with K={n_clusters}.")
    return labels


# ---------------------------------------------------------------------------
# STEP 4: Print cluster statistics
# ---------------------------------------------------------------------------

def print_cluster_statistics(df: pd.DataFrame) -> None:
    print("\nCluster Statistics:\n")
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        n_firms      = len(cluster_data)
        print(f"\nCluster {cluster_id} (Number of firms: {n_firms}):")

        for metric in CLUSTER_FEATURES:
            q1  = cluster_data[metric].quantile(0.25)
            q3  = cluster_data[metric].quantile(0.75)
            print(f"  {metric}:")
            print(f"    Min      = {cluster_data[metric].min():.4f}")
            print(f"    Q1       = {q1:.4f}")
            print(f"    Mean     = {cluster_data[metric].mean():.4f}")
            print(f"    Q3       = {q3:.4f}")
            print(f"    Max      = {cluster_data[metric].max():.4f}")
            print(f"    Std Dev  = {cluster_data[metric].std():.4f}")
            print(f"    IQR      = {(q3 - q1):.4f}")


# ---------------------------------------------------------------------------
# STEP 5: Save per-cluster CSVs
# ---------------------------------------------------------------------------

def save_cluster_csvs(df: pd.DataFrame) -> None:
    available_cols = [c for c in COLUMNS_TO_SAVE if c in df.columns]

    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df  = df[df['Cluster'] == cluster_id][available_cols]
        output_path = os.path.join(PATHS["stock_clusters"], f"cluster_{cluster_id}.csv")
        cluster_df.to_csv(output_path, index=False)
        print(f"Saved Cluster {cluster_id} ({len(cluster_df)} firms) → {output_path}")


# ---------------------------------------------------------------------------
# STEP 6: Scatter plot of clustering result
# ---------------------------------------------------------------------------

def plot_clusters(df: pd.DataFrame, n_clusters: int) -> None:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='EPS Growth (ROE x Retention)',
        y='Beta',
        hue='Cluster',
        palette='tab10',
        data=df,
        legend='full',
    )
    plt.title(f'Clustered Companies by EPS Growth and Beta  (K={n_clusters})')
    plt.xlabel('EPS Growth (ROE × Retention)')
    plt.ylabel('Beta')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    scatter_path = os.path.join(PATHS["stock_clusters"], "cluster_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Cluster scatter plot saved to: {scatter_path}")


# ---------------------------------------------------------------------------
# STEP 7: Combine all cluster CSVs into one file
# ---------------------------------------------------------------------------

def combine_cluster_csvs(cluster_folder: str, output_file: str) -> None:
    cluster_files = [
        f for f in os.listdir(cluster_folder)
        if f.startswith("cluster_") and f.endswith(".csv")
    ]
    combined_df = pd.concat(
        [pd.read_csv(os.path.join(cluster_folder, f)) for f in cluster_files],
        ignore_index=True,
    )
    combined_df.to_csv(output_file, index=False)
    print(f"Combined all clusters into: {output_file}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    # Step 1: Load data
    df, X_scaled = load_and_prepare(INPUT_CSV)

    # Step 2: Determine K
    if N_CLUSTERS is not None:
        # Manual override — skip elbow method
        k = N_CLUSTERS
        print(f"\nUsing manually specified K={k} (elbow method skipped).")
    else:
        # Run elbow method and use its recommendation
        k = run_elbow_method(X_scaled, ELBOW_K_MIN, ELBOW_K_MAX)
        print(
            f"\nTo override the elbow recommendation, set N_CLUSTERS = {k} "
            f"(or any other value) at the top of this file."
        )

    # Step 3: Fit KMeans
    df['Cluster'] = fit_kmeans(X_scaled, k)

    # Step 4: Print statistics
    print_cluster_statistics(df)

    # Step 5: Save per-cluster CSVs
    save_cluster_csvs(df)

    # Step 6: Plot clustering result
    plot_clusters(df, k)

    # Step 7: Combine into one file
    combined_output_path = os.path.join(PATHS["stock_clusters"], "all_clusters_combined.csv")
    combine_cluster_csvs(PATHS["stock_clusters"], combined_output_path)

    print(f"\nClustering complete. K={k}, {len(df['Cluster'].unique())} clusters saved.")


if __name__ == "__main__":
    main()