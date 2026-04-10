#!/usr/bin/env python3
"""
Generate disappointments table from backtest predictions.

Calculates "wasted money" for each player who underperformed expectations,
with bootstrap confidence intervals to quantify prediction uncertainty.

Wasted = min(price_paid, shortfall_war * price_per_war)

Where:
- shortfall_war = max(0, predicted_war - actual_war)
- price_per_war = total_spent / total_actual_war (from realized outcomes)

Bootstrap CIs account for model RMSE (~6 WAR), so we can distinguish
genuine overpayment from prediction noise.

Output: tabs/disappointments.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
TABS_DIR = BASE_DIR / "tabs"

MODEL_RMSE = 6.0


def bootstrap_wasted_money(pred_war, actual_war, price_cr, price_per_war, n_boot=1000, seed=42):
    """
    Bootstrap confidence intervals for wasted money estimates.

    Accounts for prediction uncertainty by adding noise to predictions
    and computing the distribution of implied waste.

    Returns dict with point estimate and 95% CI.
    """
    rng = np.random.default_rng(seed)

    wasted_samples = []
    for _ in range(n_boot):
        noise = rng.normal(0, MODEL_RMSE, size=len(pred_war))
        noisy_pred = pred_war + noise

        shortfall = np.maximum(0, noisy_pred - actual_war)
        implied_waste = shortfall * price_per_war
        wasted = np.minimum(implied_waste, price_cr)
        wasted = np.where(actual_war >= noisy_pred, 0, wasted)

        wasted_samples.append(wasted.sum())

    wasted_samples = np.array(wasted_samples)

    return {
        "total_wasted_point": np.median(wasted_samples),
        "total_wasted_ci_low": np.percentile(wasted_samples, 2.5),
        "total_wasted_ci_high": np.percentile(wasted_samples, 97.5),
    }


def bootstrap_individual_waste(pred_war, actual_war, price_cr, price_per_war, n_boot=1000, seed=42):
    """
    Bootstrap individual player wasted money with CIs.

    Returns arrays for median and CI bounds.
    """
    rng = np.random.default_rng(seed)
    n_players = len(pred_war)

    waste_matrix = np.zeros((n_boot, n_players))

    for b in range(n_boot):
        noise = rng.normal(0, MODEL_RMSE, size=n_players)
        noisy_pred = pred_war + noise

        shortfall = np.maximum(0, noisy_pred - actual_war)
        implied_waste = shortfall * price_per_war
        wasted = np.minimum(implied_waste, price_cr)
        wasted = np.where(actual_war >= noisy_pred, 0, wasted)

        waste_matrix[b, :] = wasted

    medians = np.median(waste_matrix, axis=0)
    ci_low = np.percentile(waste_matrix, 2.5, axis=0)
    ci_high = np.percentile(waste_matrix, 97.5, axis=0)

    return medians, ci_low, ci_high


def main():
    print("Generating disappointments table with bootstrap CIs...")

    df = pd.read_csv(TABS_DIR / "retroactive_predictions.csv")

    valid = df[
        df["actual_war"].notna() &
        df["predicted_war"].notna() &
        df["final_price_cr"].notna() &
        (df["final_price_cr"] > 0)
    ].copy()

    print(f"Valid predictions: {len(valid)}")

    total_spent = valid["final_price_cr"].sum()
    total_actual_war = valid["actual_war"].clip(lower=0).sum()
    PRICE_PER_WAR = total_spent / total_actual_war

    print(f"Total spent: {total_spent:.0f} Cr")
    print(f"Total actual WAR delivered: {total_actual_war:.0f}")
    print(f"Implied price per WAR: {PRICE_PER_WAR:.2f} Cr")
    print(f"Model RMSE: {MODEL_RMSE:.1f} WAR")

    valid["war_shortfall"] = (valid["predicted_war"] - valid["actual_war"]).clip(lower=0)
    valid["implied_waste_cr"] = valid["war_shortfall"] * PRICE_PER_WAR
    valid["wasted_cr"] = valid[["implied_waste_cr", "final_price_cr"]].min(axis=1)
    valid.loc[valid["actual_war"] >= valid["predicted_war"], "wasted_cr"] = 0

    print("\nComputing bootstrap confidence intervals...")
    aggregate_ci = bootstrap_wasted_money(
        valid["predicted_war"].values,
        valid["actual_war"].values,
        valid["final_price_cr"].values,
        PRICE_PER_WAR,
    )

    medians, ci_low, ci_high = bootstrap_individual_waste(
        valid["predicted_war"].values,
        valid["actual_war"].values,
        valid["final_price_cr"].values,
        PRICE_PER_WAR,
    )
    valid["wasted_cr_median"] = medians
    valid["wasted_cr_ci_low"] = ci_low
    valid["wasted_cr_ci_high"] = ci_high

    disappointments = valid[valid["wasted_cr"] > 0].copy()
    disappointments = disappointments.sort_values("wasted_cr", ascending=False)

    output = disappointments[[
        "year", "player_name", "team", "final_price_cr",
        "predicted_war", "actual_war", "war_shortfall", "wasted_cr",
        "wasted_cr_median", "wasted_cr_ci_low", "wasted_cr_ci_high"
    ]].copy()
    output.columns = [
        "year", "player", "team", "price_cr",
        "pred_war", "actual_war", "shortfall", "wasted_cr",
        "wasted_median", "wasted_ci_low", "wasted_ci_high"
    ]
    output["pct_wasted"] = (output["wasted_cr"] / output["price_cr"] * 100).round(0)

    output_path = TABS_DIR / "disappointments.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved {len(output)} disappointments to {output_path}")

    print("\n=== AGGREGATE SUMMARY (with 95% CI) ===")
    print(f"Total wasted (point): {output['wasted_cr'].sum():.0f} Cr")
    print(f"Total wasted (median): {aggregate_ci['total_wasted_point']:.0f} Cr")
    print(f"95% CI: [{aggregate_ci['total_wasted_ci_low']:.0f}, {aggregate_ci['total_wasted_ci_high']:.0f}] Cr")
    print(f"\nInterpretation: Given model RMSE of {MODEL_RMSE:.0f} WAR,")
    print(f"the 95% CI reflects uncertainty in 'true' overpayment.")

    print(f"\n=== TOP 15 (with CI) ===")
    for i, (_, row) in enumerate(output.head(15).iterrows(), 1):
        ci_str = f"[{row['wasted_ci_low']:.1f}, {row['wasted_ci_high']:.1f}]"
        print(
            f"{i:2}. {int(row['year']):4} {row['player']:<22} {row['team']:<4} | "
            f"Paid: {row['price_cr']:5.1f}Cr | "
            f"Pred: {row['pred_war']:5.1f} | "
            f"Got: {row['actual_war']:5.1f} | "
            f"Waste: {row['wasted_cr']:5.1f} {ci_str}"
        )


if __name__ == "__main__":
    main()
