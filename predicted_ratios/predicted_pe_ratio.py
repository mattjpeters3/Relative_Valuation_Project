import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config.paths import (
    STOCK_CLUSTERS_FOLDER,
    PREDICTED_PE_RATIO_RESULTS,
    ALL_SP500_STOCK_DATA_FOLDER,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SE_THRESHOLD_MULTIPLIER = 1.0
MIN_CLUSTER_SIZE = 10
MIN_INDEX_SIZE = 50
FEATURE_COLS = ['EPS Growth (ROE x Retention)', 'Beta', 'Payout Ratio']
TARGET_COL = 'PE Ratio (Current)'
SP500_CLEANED_CSV = os.path.join(ALL_SP500_STOCK_DATA_FOLDER, "cleaned_combined_sp500_data.csv")


# ---------------------------------------------------------------------------
# HELPER: Fit OLS and return diagnostics
# ---------------------------------------------------------------------------

def fit_ols_and_get_diagnostics(X_df: pd.DataFrame, y: pd.Series) -> dict:
    X_with_const = sm.add_constant(X_df, has_constant='add')
    model = sm.OLS(y, X_with_const).fit()
    residual_se = np.sqrt(model.mse_resid)
    return {
        'model':         model,
        'r_squared':     model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic':   model.fvalue,
        'f_pvalue':      model.f_pvalue,
        'coefficients':  model.params,
        'pvalues':       model.pvalues,
        'std_errors':    model.bse,
        'residual_se':   residual_se,
        'n_obs':         int(model.nobs),
        'n_predictors':  len(FEATURE_COLS),
    }


# ---------------------------------------------------------------------------
# HELPER: Format diagnostics
# ---------------------------------------------------------------------------

def format_diagnostics(diag: dict, label: str) -> str:
    lines = [
        f"\n{'=' * 60}",
        f"  REGRESSION DIAGNOSTICS — {label}",
        f"{'=' * 60}",
        f"  Observations       : {diag['n_obs']}",
        f"  Predictors         : {diag['n_predictors']}",
        f"  R²                 : {diag['r_squared']:.4f}",
        f"  Adjusted R²        : {diag['adj_r_squared']:.4f}",
        f"  F-statistic        : {diag['f_statistic']:.4f}",
        f"  F p-value          : {diag['f_pvalue']:.4f}",
        f"  Residual Std Error : {diag['residual_se']:.4f}",
        f"\n  Coefficients:",
        f"  {'Variable':<35} {'Coef':>10} {'Std Err':>10} {'p-value':>10} {'Sig':>5}",
        f"  {'-' * 72}",
    ]
    for var in diag['coefficients'].index:
        coef = diag['coefficients'][var]
        se   = diag['std_errors'][var]
        pval = diag['pvalues'][var]
        sig  = _significance_stars(pval)
        lines.append(f"  {var:<35} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig:>5}")
    lines += [
        f"  {'-' * 72}",
        f"  Significance: *** p<0.01  ** p<0.05  * p<0.10",
        f"{'=' * 60}\n",
    ]
    return '\n'.join(lines)


def _significance_stars(pval: float) -> str:
    if pval < 0.01:  return '***'
    if pval < 0.05:  return '**'
    if pval < 0.10:  return '*'
    return ''


# ---------------------------------------------------------------------------
# HELPER: LOO prediction loop
# ---------------------------------------------------------------------------

def run_loo_predictions(working_df: pd.DataFrame, store_diagnostics: bool = False):
    """
    Run LOO predictions for every firm in working_df.

    If store_diagnostics=True, also capture each firm's individual LOO
    regression output (coefficients, p-values, std errors, R², F p-value,
    residual SE) so they can be displayed in the stock lookup page.

    Returns:
        predicted_pe_list          — always returned
        loo_diagnostics (dict)     — only when store_diagnostics=True,
                                     keyed by the Ticker value for that row
    """
    predicted_pe_list = []
    loo_diagnostics   = {}

    for idx in working_df.index:
        target_features = working_df.loc[idx, FEATURE_COLS]
        ticker          = working_df.loc[idx, 'Ticker'] if 'Ticker' in working_df.columns else str(idx)
        train_df        = working_df.drop(index=idx)

        if len(train_df) < len(FEATURE_COLS) + 2:
            predicted_pe_list.append(np.nan)
            continue

        X_train_np = np.column_stack([np.ones(len(train_df)), train_df[FEATURE_COLS].values])
        y_train_np = train_df[TARGET_COL].values

        try:
            loo_model   = sm.OLS(y_train_np, X_train_np).fit()
            X_target_np = np.array([1.0] + list(target_features.values))
            predicted_pe_list.append(float(loo_model.predict(X_target_np)[0]))

            if store_diagnostics:
                feat_names = FEATURE_COLS
                params     = loo_model.params       # [const, f1, f2, f3]
                pvals      = loo_model.pvalues
                bse        = loo_model.bse
                loo_diagnostics[ticker] = {
                    'n_obs':       int(loo_model.nobs),
                    'r_squared':   round(float(loo_model.rsquared),     4),
                    'adj_r_squared': round(float(loo_model.rsquared_adj), 4),
                    'f_statistic': round(float(loo_model.fvalue),       4),
                    'f_pvalue':    round(float(loo_model.f_pvalue),     4),
                    'residual_se': round(float(np.sqrt(loo_model.mse_resid)), 4),
                    'const':       round(float(params[0]), 4),
                    'coefficients': [
                        {
                            'variable': feat_names[i],
                            'coef':    round(float(params[i + 1]), 4),
                            'std_err': round(float(bse[i + 1]),    4),
                            'pvalue':  round(float(pvals[i + 1]),  4),
                        }
                        for i in range(len(feat_names))
                    ],
                }
        except Exception as e:
            print(f"  LOO prediction failed for index {idx}: {type(e).__name__}: {e}")
            predicted_pe_list.append(np.nan)

    if store_diagnostics:
        return predicted_pe_list, loo_diagnostics
    return predicted_pe_list


# ---------------------------------------------------------------------------
# HELPER: Valuation signal
# ---------------------------------------------------------------------------

def assign_valuation_signal(actual_pe, predicted_pe, residual_se,
                             multiplier=SE_THRESHOLD_MULTIPLIER):
    if pd.isna(actual_pe) or pd.isna(predicted_pe):
        return 'N/A'
    diff = actual_pe - predicted_pe
    band = multiplier * residual_se
    if diff < -band:  return 'Undervalued'
    if diff > band:   return 'Overvalued'
    return 'Fairly Valued'


# ---------------------------------------------------------------------------
# HELPER: Derive Payout Ratio
# ---------------------------------------------------------------------------

def derive_payout_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if 'Payout Ratio' not in df.columns:
        eps_growth = df['EPS Growth (ROE x Retention)'].replace(0, np.nan)
        df['Payout Ratio'] = (
            df['Dividend Yield'] / eps_growth
        ).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ---------------------------------------------------------------------------
# STEP 1: Per-cluster regression with LOO predictions
# ---------------------------------------------------------------------------

def calculate_predicted_pe_ratios():
    cluster_files = [
        f for f in os.listdir(STOCK_CLUSTERS_FOLDER)
        if f.endswith('.csv') and not f.startswith('all_clusters')
    ]
    if not cluster_files:
        print("No cluster CSV files found.")
        return

    for file in sorted(cluster_files):
        cluster_df   = pd.read_csv(os.path.join(STOCK_CLUSTERS_FOLDER, file))
        cluster_name = file.replace('.csv', '')
        cluster_df   = derive_payout_ratio(cluster_df)

        required_cols = FEATURE_COLS + [TARGET_COL]
        working_df = (
            cluster_df.dropna(subset=required_cols)
            .copy()
            .reset_index(drop=True)
        )

        if len(working_df) < MIN_CLUSTER_SIZE:
            print(f"Skipping {file} — only {len(working_df)} complete rows "
                  f"(minimum required: {MIN_CLUSTER_SIZE}).")
            continue

        diag = fit_ols_and_get_diagnostics(working_df[FEATURE_COLS], working_df[TARGET_COL])
        print(format_diagnostics(diag, cluster_name))
        residual_se = diag['residual_se']

        if diag['f_pvalue'] > 0.10:
            print(f"  WARNING: {cluster_name} regression is not statistically significant "
                  f"(F p-value = {diag['f_pvalue']:.4f}). Signals should be treated with caution.\n")
        if diag['r_squared'] < 0.10:
            print(f"  WARNING: {cluster_name} has very low R² ({diag['r_squared']:.4f}). "
                  f"The model explains little PE variation in this cluster.\n")

        model_is_significant = diag['f_pvalue'] <= 0.10

        # Run LOO with per-firm diagnostic capture
        predicted_pe_list, loo_diags = run_loo_predictions(working_df, store_diagnostics=True)

        # Save per-firm LOO diagnostics to JSON so the dashboard can look them up
        import json
        loo_json_path = os.path.join(PREDICTED_PE_RATIO_RESULTS, f"loo_diagnostics_{cluster_name}.json")
        with open(loo_json_path, 'w') as f:
            json.dump(loo_diags, f, indent=2)

        working_df['Predicted PE (Cluster)']     = np.round(predicted_pe_list, 2)
        working_df['PE Difference (Cluster)']    = np.round(
            working_df[TARGET_COL] - working_df['Predicted PE (Cluster)'], 2)
        working_df['Valuation Signal (Cluster)'] = working_df.apply(
            lambda row: assign_valuation_signal(
                row[TARGET_COL], row['Predicted PE (Cluster)'], residual_se
            ) if model_is_significant else 'Model Insignificant',
            axis=1,
        )
        working_df['Cluster R²']          = round(diag['r_squared'],     4)
        working_df['Cluster Adj R²']      = round(diag['adj_r_squared'], 4)
        working_df['Cluster F p-value']   = round(diag['f_pvalue'],      4)
        working_df['Cluster Residual SE'] = round(residual_se,           4)

        output_file = os.path.join(PREDICTED_PE_RATIO_RESULTS, f"predicted_{file}")
        working_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}\n")


# ---------------------------------------------------------------------------
# STEP 2: Whole-index regression with LOO predictions
# ---------------------------------------------------------------------------

def calculate_whole_index_pe_ratios():
    if not os.path.exists(SP500_CLEANED_CSV):
        print(f"Whole-index CSV not found at: {SP500_CLEANED_CSV}")
        return

    index_df = pd.read_csv(SP500_CLEANED_CSV)
    index_df = derive_payout_ratio(index_df)

    required_cols = FEATURE_COLS + [TARGET_COL, 'Ticker']
    index_df = (
        index_df.dropna(subset=required_cols)
        .copy()
        .reset_index(drop=True)
    )

    if len(index_df) < MIN_INDEX_SIZE:
        print(f"Whole-index regression skipped — only {len(index_df)} complete rows.")
        return

    print(f"\nRunning whole-index regression on {len(index_df)} firms...")
    diag = fit_ols_and_get_diagnostics(index_df[FEATURE_COLS], index_df[TARGET_COL])
    print(format_diagnostics(diag, "WHOLE INDEX"))
    residual_se = diag['residual_se']

    if diag['f_pvalue'] > 0.10:
        print(f"  WARNING: Whole-index regression is not statistically significant "
              f"(F p-value = {diag['f_pvalue']:.4f}).\n")
    if diag['r_squared'] < 0.10:
        print(f"  WARNING: Whole-index R² is very low ({diag['r_squared']:.4f}).\n")

    model_is_significant = diag['f_pvalue'] <= 0.10

    # Run LOO with per-firm diagnostic capture for the index model
    import json
    predicted_pe_list, loo_diags_index = run_loo_predictions(index_df, store_diagnostics=True)
    loo_json_path = os.path.join(PREDICTED_PE_RATIO_RESULTS, "loo_diagnostics_index.json")
    with open(loo_json_path, 'w') as f:
        json.dump(loo_diags_index, f, indent=2)

    index_df['Predicted PE (Index)']     = np.round(predicted_pe_list, 2)
    index_df['PE Difference (Index)']    = np.round(
        index_df[TARGET_COL] - index_df['Predicted PE (Index)'], 2)
    index_df['Valuation Signal (Index)'] = index_df.apply(
        lambda row: assign_valuation_signal(
            row[TARGET_COL], row['Predicted PE (Index)'], residual_se
        ) if model_is_significant else 'Model Insignificant',
        axis=1,
    )
    index_df['Index R²']          = round(diag['r_squared'],     4)
    index_df['Index Adj R²']      = round(diag['adj_r_squared'], 4)
    index_df['Index F p-value']   = round(diag['f_pvalue'],      4)
    index_df['Index Residual SE'] = round(residual_se,           4)

    output_path = os.path.join(PREDICTED_PE_RATIO_RESULTS, "whole_index_predictions.csv")
    index_df.to_csv(output_path, index=False)
    print(f"  Saved whole-index predictions to: {output_path}\n")


# ---------------------------------------------------------------------------
# STEP 3: Combine cluster + index signals into master table
# ---------------------------------------------------------------------------

def combine_and_filter_results():
    cluster_dfs = []
    for file in sorted(os.listdir(PREDICTED_PE_RATIO_RESULTS)):
        if file.startswith('predicted_') and file.endswith('.csv'):
            df = pd.read_csv(os.path.join(PREDICTED_PE_RATIO_RESULTS, file))
            df['Source Cluster'] = file.replace('predicted_', '').replace('.csv', '')
            cluster_dfs.append(df)

    if not cluster_dfs:
        print("No cluster prediction files found.")
        return

    cluster_combined = pd.concat(cluster_dfs, ignore_index=True)
    cluster_combined = cluster_combined.drop_duplicates(subset=['Ticker'])

    # Keep needed cluster columns (include Sector if present)
    cluster_cols = [
        'Ticker', 'Predicted PE (Cluster)', 'PE Difference (Cluster)',
        'Valuation Signal (Cluster)', 'Cluster R²', 'Cluster Adj R²',
        'Cluster F p-value', 'Cluster Residual SE', 'Source Cluster',
        'Sector',
    ]
    cluster_cols = [c for c in cluster_cols if c in cluster_combined.columns]
    cluster_combined = cluster_combined[cluster_cols]

    # Load whole-index predictions
    index_path = os.path.join(PREDICTED_PE_RATIO_RESULTS, "whole_index_predictions.csv")
    if not os.path.exists(index_path):
        print("Whole-index predictions file not found — skipping merge.")
        master_df = cluster_combined
    else:
        index_df = pd.read_csv(index_path)
        index_cols = [
            'Ticker', TARGET_COL,
            'EPS Growth (ROE x Retention)', 'Beta', 'Payout Ratio',
            'Predicted PE (Index)', 'PE Difference (Index)',
            'Valuation Signal (Index)',
            'Index R²', 'Index Adj R²', 'Index F p-value', 'Index Residual SE',
        ]
        # Drop Sector from index side to avoid duplication (already in cluster side)
        index_cols = [c for c in index_cols if c != 'Sector']
        index_cols = [c for c in index_cols if c in index_df.columns]
        index_df   = index_df[index_cols]

        master_df = cluster_combined.merge(index_df, on='Ticker', how='left')

        def combined_signal(row):
            c = str(row.get('Valuation Signal (Cluster)', '')).strip()
            i = str(row.get('Valuation Signal (Index)',   '')).strip()
            if 'N/A' in (c, i) or 'Insufficient' in (c + i):
                return 'Insufficient Data'
            if c == 'Model Insignificant' or i == 'Model Insignificant':
                if c != 'Model Insignificant' and i == 'Model Insignificant':
                    return f'{c} (Cluster only)'
                if i != 'Model Insignificant' and c == 'Model Insignificant':
                    return f'{i} (Index only)'
                return 'Model Insignificant'
            if c == i:
                if c == 'Undervalued': return 'Strong Undervalued'
                if c == 'Overvalued':  return 'Strong Overvalued'
                return 'Fairly Valued'
            return 'Conflicting'

        master_df['Combined Signal'] = master_df.apply(combined_signal, axis=1)

    # Summary
    print("\n" + "=" * 60)
    print("  MASTER VALUATION SUMMARY")
    print("=" * 60)
    sig_col = 'Combined Signal' if 'Combined Signal' in master_df.columns else 'Valuation Signal (Cluster)'
    print(master_df[sig_col].value_counts().to_string())
    print(f"\n  Total firms in master table: {len(master_df)}")
    print("=" * 60 + "\n")

    master_path = os.path.join(PREDICTED_PE_RATIO_RESULTS, "master_valuations.csv")
    master_df.to_csv(master_path, index=False)
    print(f"Saved master valuation table to: {master_path}")

    priority_signals = [
        'Strong Undervalued', 'Strong Overvalued',
        'Undervalued (Cluster only)', 'Overvalued (Cluster only)',
        'Undervalued (Index only)',   'Overvalued (Index only)',
    ]
    for signal in priority_signals:
        signal_df = master_df[master_df[sig_col] == signal].copy()
        if signal_df.empty:
            continue
        if 'Undervalued' in signal:
            signal_df = signal_df.sort_values('PE Difference (Cluster)', ascending=True)
        elif 'Overvalued' in signal:
            signal_df = signal_df.sort_values('PE Difference (Cluster)', ascending=False)
        filename = signal.lower().replace(' ', '_').replace('(', '').replace(')', '') + '_firms.csv'
        signal_df.to_csv(os.path.join(PREDICTED_PE_RATIO_RESULTS, filename), index=False)
        print(f"Saved {len(signal_df):>3} {signal} firms to: {filename}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Step 1: Running per-cluster regressions...")
    calculate_predicted_pe_ratios()

    print("\nStep 2: Running whole-index regression...")
    calculate_whole_index_pe_ratios()

    print("\nStep 3: Merging signals into master valuation table...")
    combine_and_filter_results()