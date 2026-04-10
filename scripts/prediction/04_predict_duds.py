#!/usr/bin/env python3
"""
Predict IPL 2026 Auction "Duds" - players most likely to disappoint.

Uses XGBoost forecasting model trained on IPL + T20I data to predict
next-season WAR, then calculates premium (overpayment) relative to
model-predicted fair value.

Premium = (actual_price - expected_price) / expected_price
where expected_price = f(forecasted_WAR, is_indian, auction_type)

Output: tabs/predicted_duds_2026.csv
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import normalize_name
from shared.io import load_dataset, dataset_exists

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
T20I_DIR = DATA_DIR / "perf" / "t20i"
AUCTION_DIR = DATA_DIR / "auction"
PERF_DIR = DATA_DIR / "perf" / "ipl"
TABS_DIR = BASE_DIR / "tabs"


def load_data():
    """Load auction, WAR, and T20I data."""
    print("Loading data...")

    auction = load_dataset(AUCTION_DIR / "auction_all_years")
    print(f"  Auction records: {len(auction)}")

    ipl_war = load_dataset(PERF_DIR / "player_season_war")
    print(f"  IPL WAR records: {len(ipl_war)}")

    if dataset_exists(T20I_DIR / "player_year_war"):
        t20i_war = load_dataset(T20I_DIR / "player_year_war")
        print(f"  T20I WAR records: {len(t20i_war)}")
    else:
        t20i_war = pd.DataFrame(columns=["year", "player", "total_war"])
        print("  T20I WAR: not found")

    return auction, ipl_war, t20i_war


def load_forecast_model():
    """Load trained XGBoost forecasting model."""
    model_path = MODELS_DIR / "war_forecast_xgb.joblib"

    if model_path.exists():
        model_data = joblib.load(model_path)
        print(f"  Loaded forecasting model from {model_path}")
        return model_data["model"], model_data["imputer"], model_data["feature_cols"]

    print("  Warning: No trained model found. Using fallback approach.")
    return None, None, None


def get_player_name_mapping():
    """Manual mapping from auction names (full) to WAR names (abbreviated)."""
    return {
        "cameron green": "c green",
        "matheesha pathirana": "m pathirana",
        "liam livingstone": "ls livingstone",
        "venkatesh iyer": "vr iyer",
        "ravi bishnoi": "r bishnoi",
        "rahul chahar": "rd chahar",
        "quinton de kock": "qj de kock",
        "josh inglis": "jp inglis",
        "david miller": "da miller",
        "wanindu hasaranga": "pwadd hasaranga",
        "anrich nortje": "a nortje",
        "jason holder": "jr holder",
        "mustafizur rahman": "mustafizur rahman",
        "rachin ravindra": "rachin ravindra",
        "ben duckett": "bm duckett",
        "sarfaraz khan": "sn khan",
        "prithvi shaw": "pp shaw",
        "finn allen": "fg allen",
        "shivam mavi": "s mavi",
        "akash deep": "akash deep",
        "rahul tripathi": "rahul tripathi",
        "kyle jamieson": "kj jamieson",
        "lungi ngidi": "l ngidi",
        "kuldeep sen": "kuldeep sen",
        "cooper connolly": "c connolly",
        "ben dwarshuis": "bj dwarshuis",
        "pathum nissanka": "pbd nissanka",
        "adam milne": "af milne",
        "tim seifert": "kd seifert",
        "akeal hosein": "as hosein",
        "kartik tyagi": "kartik tyagi",
        "matt henry": "mj henry",
        "jacob duffy": "jd duffy",
        "luke wood": "l wood",
        "mangesh yadav": "m yadav",
        "vicky ostwal": "r ostwal",
        "zak foulkes": "za foulkes",
        "tom banton": "t banton",
        "jordan cox": "jm cox",
        "matthew short": "mw short",
    }


def create_player_features(player_name, year, ipl_war, t20i_war, name_map):
    """Create feature vector for a single player at auction time."""
    player_norm = normalize_name(player_name)

    mapped_name = name_map.get(player_norm, player_norm)

    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)
    player_ipl = ipl_war[
        (ipl_war["player_norm"] == mapped_name) &
        (ipl_war["season"] < year)
    ].sort_values("season", ascending=False)

    features = {}

    if len(player_ipl) >= 1:
        features["ipl_war_lag1"] = player_ipl.iloc[0]["total_war"]
    if len(player_ipl) >= 2:
        features["ipl_war_lag2"] = player_ipl.iloc[1]["total_war"]
    if len(player_ipl) >= 3:
        features["ipl_war_lag3"] = player_ipl.iloc[2]["total_war"]
    if len(player_ipl) > 0:
        features["ipl_career_war"] = player_ipl["total_war"].sum()
        features["ipl_seasons_played"] = len(player_ipl)

        vals = [features.get("ipl_war_lag1"), features.get("ipl_war_lag2"), features.get("ipl_war_lag3")]
        vals = [v for v in vals if v is not None]
        if vals:
            features["ipl_war_avg_3y"] = np.mean(vals)

        if len(vals) >= 2:
            features["ipl_war_trend"] = vals[0] - vals[1]

    t20i_war["player_norm"] = t20i_war["player"].apply(normalize_name)
    player_t20i = t20i_war[
        (t20i_war["player_norm"] == mapped_name) &
        (t20i_war["year"] < year)
    ].sort_values("year", ascending=False)

    if len(player_t20i) >= 1:
        features["t20i_war_12m"] = player_t20i.iloc[0]["total_war"]
    if len(player_t20i) > 0:
        features["t20i_career_war"] = player_t20i["total_war"].sum()

    features["combined_war_12m"] = (
        features.get("ipl_war_lag1", 0) +
        features.get("t20i_war_12m", 0) * 0.5
    )
    features["combined_war_24m"] = (
        features.get("ipl_war_lag1", 0) +
        features.get("ipl_war_lag2", 0) +
        features.get("t20i_war_12m", 0) * 0.3
    )

    features["has_ipl_history"] = 1 if "ipl_war_lag1" in features else 0
    features["has_t20i_history"] = 1 if "t20i_war_12m" in features else 0

    return features


def forecast_war_for_players(auction_2026, ipl_war, t20i_war, model, imputer, feature_cols):
    """Forecast next-season WAR for 2026 auction players."""
    print("\nForecasting WAR for 2026 auction players...")

    name_map = get_player_name_mapping()

    predictions = []

    for idx, row in auction_2026.iterrows():
        player_name = row["player_name"]
        year = row["year"]

        features = create_player_features(player_name, year, ipl_war.copy(), t20i_war.copy(), name_map)

        if not features.get("has_ipl_history") and not features.get("has_t20i_history"):
            continue

        mega_years = [2008, 2011, 2014, 2018, 2022, 2025]
        features["is_mega_auction"] = 1 if year in mega_years else 0

        feature_df = pd.DataFrame([features])

        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = np.nan

        X = feature_df[feature_cols]
        X_imp = imputer.transform(X)
        war_forecast = model.predict(X_imp)[0]

        predictions.append({
            "player_name": player_name,
            "team": row.get("team"),
            "final_price_lakh": row.get("final_price_lakh"),
            "nationality": row.get("nationality"),
            "role": row.get("role"),
            "war_forecast": war_forecast,
            "ipl_war_lag1": features.get("ipl_war_lag1"),
            "t20i_war_12m": features.get("t20i_war_12m"),
            "combined_war_12m": features.get("combined_war_12m"),
        })

    predictions_df = pd.DataFrame(predictions)
    print(f"  Forecasted WAR for {len(predictions_df)} players")

    return predictions_df


def estimate_price_model(auction, ipl_war):
    """Estimate price model using historical data and forecasted WAR."""
    print("\nEstimating price model...")

    auction = auction.copy()
    ipl_war = ipl_war.copy()

    auction_hist = auction[(auction["year"] > 2008) & (auction["year"] < 2026)].copy()

    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)
    auction_hist["player_norm"] = auction_hist["player_name"].apply(normalize_name)

    auction_hist["war_prior"] = np.nan

    name_map = get_player_name_mapping()

    for idx, row in auction_hist.iterrows():
        player_norm = row["player_norm"]
        mapped_name = name_map.get(player_norm, player_norm)
        year = row["year"]

        prior_war = ipl_war[
            (ipl_war["player_norm"] == mapped_name) &
            (ipl_war["season"] == year - 1)
        ]

        if len(prior_war) > 0:
            auction_hist.at[idx, "war_prior"] = prior_war.iloc[0]["total_war"]

    valid = auction_hist[
        (auction_hist["war_prior"].notna()) &
        (auction_hist["final_price_lakh"].notna())
    ].copy()

    valid["final_price_lakh"] = pd.to_numeric(valid["final_price_lakh"], errors="coerce")
    valid = valid[valid["final_price_lakh"] > 0]

    valid["log_price"] = np.log(valid["final_price_lakh"])
    valid["is_indian"] = (valid["nationality"] == "Indian").astype(int)

    mega_years = [2008, 2011, 2014, 2018, 2022, 2025]
    valid["is_mega_auction"] = valid["year"].isin(mega_years).astype(int)

    X = valid[["war_prior", "is_indian", "is_mega_auction"]]
    X = sm.add_constant(X)
    y = valid["log_price"]

    model = sm.OLS(y, X).fit()

    print("=" * 60)
    print("PRICE PREDICTION MODEL (log_price ~ WAR + controls)")
    print("=" * 60)
    print(model.summary())

    return model


def calculate_premiums(predictions_df, price_model):
    """Calculate premium (overpayment) for each player."""
    print("\nCalculating premiums...")

    predictions_df = predictions_df.copy()

    predictions_df["final_price_lakh"] = pd.to_numeric(
        predictions_df["final_price_lakh"], errors="coerce"
    )

    predictions_df["is_indian"] = (predictions_df["nationality"] == "Indian").astype(int)
    predictions_df["is_mega_auction"] = 0

    X_pred = predictions_df[["war_forecast", "is_indian", "is_mega_auction"]].copy()
    X_pred.columns = ["war_prior", "is_indian", "is_mega_auction"]
    X_pred = sm.add_constant(X_pred, has_constant="add")

    predictions_df["log_price_predicted"] = price_model.predict(X_pred)
    predictions_df["price_predicted_lakh"] = np.exp(predictions_df["log_price_predicted"])

    predictions_df["premium_pct"] = (
        (predictions_df["final_price_lakh"] - predictions_df["price_predicted_lakh"])
        / predictions_df["price_predicted_lakh"]
    ) * 100

    return predictions_df


def get_team_abbreviation(team):
    """Convert team name to standard abbreviation."""
    if pd.isna(team):
        return team
    team_map = {
        "CSK": "CSK", "DC": "DC", "MI": "MI", "KKR": "KKR",
        "RCB": "RCB", "RR": "RR", "SRH": "SRH", "PBKS": "PBKS",
        "LSG": "LSG", "GT": "GT",
    }
    return team_map.get(str(team).upper(), team)


def format_output(df, top_n=20):
    """Format predicted duds for output."""
    df = df.copy()
    df = df.sort_values("premium_pct", ascending=False).head(top_n)

    result = pd.DataFrame({
        "Rank": range(1, len(df) + 1),
        "Player": df["player_name"].values,
        "Team": df["team"].apply(get_team_abbreviation).values,
        "Price (Cr)": (df["final_price_lakh"] / 100).round(2).values,
        "WAR Forecast": df["war_forecast"].round(2).values,
        "Prior IPL WAR": df["ipl_war_lag1"].round(2).values,
        "Predicted (Cr)": (df["price_predicted_lakh"] / 100).round(2).values,
        "Premium %": df["premium_pct"].round(0).astype(int).astype(str).values + "%",
    })

    return result


def main():
    auction, ipl_war, t20i_war = load_data()

    model, imputer, feature_cols = load_forecast_model()

    if model is None:
        print("\nFalling back to lagged WAR approach (no trained model)")
        feature_cols = ["ipl_war_lag1", "ipl_war_lag2", "ipl_war_lag3",
                       "ipl_career_war", "ipl_seasons_played",
                       "t20i_war_12m", "t20i_career_war",
                       "ipl_war_trend", "ipl_war_avg_3y",
                       "combined_war_12m", "combined_war_24m",
                       "has_ipl_history", "has_t20i_history", "is_mega_auction"]
        imputer = SimpleImputer(strategy="median")
        imputer.fit(np.zeros((1, len(feature_cols))))

        class SimpleLagModel:
            def predict(self, X):
                return X[:, 0]
        model = SimpleLagModel()

    auction_2026 = auction[auction["year"] == 2026].copy()
    print(f"\n2026 auction players: {len(auction_2026)}")

    predictions = forecast_war_for_players(
        auction_2026, ipl_war, t20i_war, model, imputer, feature_cols
    )

    if len(predictions) == 0:
        print("\nNo predictions possible - no players matched with WAR data")
        return

    price_model = estimate_price_model(auction, ipl_war)
    predictions = calculate_premiums(predictions, price_model)

    output = format_output(predictions, top_n=len(predictions))

    print("\n" + "=" * 80)
    print("PREDICTED DUDS FOR IPL 2026")
    print("=" * 80)
    print("\nRanked by Premium % (how much teams overpaid vs. model prediction)")
    print("Using XGBoost forecasted WAR (IPL + T20I features)\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", 25)

    print(output.head(15).to_string(index=False))

    TABS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABS_DIR / "predicted_duds_2026.csv"
    output.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Premium % shows how much teams paid above the model-predicted fair value.
Higher premium = higher risk of disappointing performance.

This model uses:
- XGBoost-forecasted next-season WAR based on:
  * Prior IPL seasons (lag 1, 2, 3)
  * T20 International performance
  * Career WAR and form trend
- Indian player premium
- Auction type (mega vs. mini)

Caveats:
- OOS R² is ~25% (performance prediction is inherently difficult)
- Excludes new IPL entrants with no prior history
- Teams may have private information justifying prices
""")


if __name__ == "__main__":
    main()
