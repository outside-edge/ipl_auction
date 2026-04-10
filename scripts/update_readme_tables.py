#!/usr/bin/env python3
"""
Update README.md with tables from CSV data.

Replaces content between marker comments like:
<!-- TABLE:disappointments:start -->
...table content...
<!-- TABLE:disappointments:end -->
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
README_PATH = BASE_DIR / "readme.md"
TABS_DIR = BASE_DIR / "tabs"


def csv_to_markdown_table(df, max_rows=15):
    """Convert DataFrame to markdown table string."""
    df = df.head(max_rows)

    # Header
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"

    # Rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(v) for v in row.values) + " |"
        rows.append(row_str)

    return "\n".join([header, separator] + rows)


def generate_disappointments_table():
    """Generate top disappointments table."""
    df = pd.read_csv(TABS_DIR / "disappointments.csv")

    # Format for display
    display = df.head(15).copy()
    display["year"] = display["year"].astype(int)
    display["price_cr"] = display["price_cr"].apply(lambda x: f"₹{x:.1f}Cr")
    display["pred_war"] = display["pred_war"].apply(lambda x: f"{x:.1f}")
    display["actual_war"] = display["actual_war"].apply(lambda x: f"{x:.1f}")
    display["wasted_cr"] = display["wasted_cr"].apply(lambda x: f"₹{x:.1f}Cr")
    display["pct_wasted"] = display["pct_wasted"].apply(lambda x: f"{x:.0f}%")

    display = display[["year", "player", "team", "price_cr", "pred_war", "actual_war", "wasted_cr"]]
    display.columns = ["Year", "Player", "Team", "Paid", "Predicted", "Actual", "Wasted"]

    # Add summary
    total_wasted = pd.read_csv(TABS_DIR / "disappointments.csv")["wasted_cr"].sum()
    n_players = len(pd.read_csv(TABS_DIR / "disappointments.csv"))

    summary = f"**Total wasted: ₹{total_wasted:.0f} Cr** across {n_players} disappointing players\n\n"

    return summary + csv_to_markdown_table(display)


def generate_team_efficiency_table():
    """Generate team efficiency table."""
    df = pd.read_csv(TABS_DIR / "team_efficiency.csv")

    # Filter to teams with 20+ players
    df = df[df["n_players"] >= 20].copy()

    display = df[["team", "n_players", "total_spent_cr", "total_actual_war", "war_per_cr"]].copy()
    display["total_spent_cr"] = display["total_spent_cr"].apply(lambda x: f"₹{x:.0f}Cr")
    display["total_actual_war"] = display["total_actual_war"].apply(lambda x: f"{x:.0f}")
    display["war_per_cr"] = display["war_per_cr"].apply(lambda x: f"{x:.2f}")

    display.columns = ["Team", "N", "Spent", "WAR", "WAR/Cr"]

    return csv_to_markdown_table(display)


def generate_backtest_summary_table():
    """Generate backtest summary table."""
    df = pd.read_csv(TABS_DIR / "retroactive_summary.csv")

    display = df[["year", "n_players", "r2", "rank_corr", "rmse"]].copy()
    display["year"] = display["year"].astype(int)
    display["r2"] = display["r2"].apply(lambda x: f"{x:.2f}")
    display["rank_corr"] = display["rank_corr"].apply(lambda x: f"{x:.2f}")
    display["rmse"] = display["rmse"].apply(lambda x: f"{x:.1f}")

    display.columns = ["Year", "N", "R²", "Rank ρ", "RMSE"]

    return csv_to_markdown_table(display)


def update_readme():
    """Update README with all tables."""
    readme = README_PATH.read_text()

    tables = {
        "disappointments": generate_disappointments_table,
        "team_efficiency": generate_team_efficiency_table,
        "backtest_summary": generate_backtest_summary_table,
    }

    for name, generator in tables.items():
        start_marker = f"<!-- TABLE:{name}:start -->"
        end_marker = f"<!-- TABLE:{name}:end -->"

        if start_marker in readme and end_marker in readme:
            try:
                table_content = generator()

                start_idx = readme.index(start_marker) + len(start_marker)
                end_idx = readme.index(end_marker)

                readme = readme[:start_idx] + "\n" + table_content + "\n" + readme[end_idx:]
                print(f"Updated table: {name}")
            except Exception as e:
                print(f"Error generating {name}: {e}")
        else:
            print(f"Markers not found for: {name}")

    README_PATH.write_text(readme)
    print(f"Wrote {README_PATH}")


if __name__ == "__main__":
    update_readme()
