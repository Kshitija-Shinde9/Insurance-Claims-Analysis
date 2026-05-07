"""Generate EDA visuals for the freMTPL2 final project."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fremtpl2_utils import OUTPUT_DIR, ensure_directories, load_modeling_data, write_text


EDA_DIR = OUTPUT_DIR / "eda"


def savefig(name: str) -> None:
    path = EDA_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def weighted_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            Policies=("IDpol", "count"),
            Exposure=("Exposure", "sum"),
            Claims=("ClaimNb", "sum"),
            TotalClaimAmount=("TotalClaimAmount", "sum"),
        )
        .reset_index()
    )
    grouped["ClaimFrequency"] = grouped["Claims"] / grouped["Exposure"]
    grouped["PurePremium"] = grouped["TotalClaimAmount"] / grouped["Exposure"]
    return grouped


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid")
    df = load_modeling_data()

    sample = df.sample(min(len(df), 120_000), random_state=42)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="ClaimNb", color="#3b82f6")
    plt.title("Claim Count Distribution")
    plt.xlabel("Number of Claims")
    plt.ylabel("Policy Count")
    savefig("01_claim_count_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(sample["Exposure"], bins=50, color="#14b8a6")
    plt.title("Policy Exposure Distribution")
    plt.xlabel("Exposure")
    plt.ylabel("Policy Count")
    savefig("02_exposure_distribution.png")

    positive_claims = df[df["TotalClaimAmount"] > 0].copy()
    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(positive_claims["TotalClaimAmount"]), bins=60, color="#f97316")
    plt.title("Log Total Claim Amount Distribution - Claim Policies Only")
    plt.xlabel("log(1 + Total Claim Amount)")
    plt.ylabel("Policy Count")
    savefig("03_log_claim_amount_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(sample["PurePremium"]), bins=70, color="#8b5cf6")
    plt.title("Log Pure Premium Distribution")
    plt.xlabel("log(1 + Pure Premium)")
    plt.ylabel("Policy Count")
    savefig("04_log_pure_premium_distribution.png")

    age = weighted_summary(df, "DriverAgeBand")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=age, x="DriverAgeBand", y="ClaimFrequency", color="#2563eb")
    plt.title("Claim Frequency by Driver Age Band")
    plt.xlabel("Driver Age Band")
    plt.ylabel("Claims per Exposure Unit")
    savefig("05_claim_frequency_by_driver_age.png")

    veh_age = weighted_summary(df, "VehicleAgeBand")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=veh_age, x="VehicleAgeBand", y="ClaimFrequency", color="#0f766e")
    plt.title("Claim Frequency by Vehicle Age Band")
    plt.xlabel("Vehicle Age Band")
    plt.ylabel("Claims per Exposure Unit")
    savefig("06_claim_frequency_by_vehicle_age.png")

    bm = weighted_summary(df, "BonusMalusBand")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=bm, x="BonusMalusBand", y="PurePremium", color="#dc2626")
    plt.title("Pure Premium by Bonus-Malus Band")
    plt.xlabel("Bonus-Malus Band")
    plt.ylabel("Pure Premium")
    savefig("07_pure_premium_by_bonus_malus.png")

    region = weighted_summary(df, "Region").sort_values("PurePremium", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=region.head(15), y="Region", x="PurePremium", color="#7c3aed")
    plt.title("Top Regions by Pure Premium")
    plt.xlabel("Pure Premium")
    plt.ylabel("Region")
    savefig("08_top_regions_by_pure_premium.png")

    area = weighted_summary(df, "Area").sort_values("Area")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=area, x="Area", y="PurePremium", marker="o", linewidth=2.5)
    plt.title("Pure Premium by Area Density Category")
    plt.xlabel("Area")
    plt.ylabel("Pure Premium")
    savefig("09_pure_premium_by_area.png")

    gas = weighted_summary(df, "VehGas")
    plt.figure(figsize=(7, 5))
    sns.barplot(data=gas, x="VehGas", y="ClaimFrequency", color="#0891b2")
    plt.title("Claim Frequency by Vehicle Gas Type")
    plt.xlabel("Vehicle Gas")
    plt.ylabel("Claims per Exposure Unit")
    savefig("10_claim_frequency_by_gas_type.png")

    summary = f"""EDA summary

Rows analyzed: {len(df):,}
Columns: {len(df.columns):,}
Policies with no claim count: {(df["ClaimNb"] == 0).sum():,}
Policies with at least one claim count: {(df["ClaimNb"] > 0).sum():,}
Share with no claim count: {(df["ClaimNb"] == 0).mean():.4%}
Total exposure: {df["Exposure"].sum():,.2f}
Total claim amount: {df["TotalClaimAmount"].sum():,.2f}
Mean pure premium: {df["PurePremium"].mean():,.4f}
Weighted pure premium: {df["TotalClaimAmount"].sum() / df["Exposure"].sum():,.4f}

Generated EDA figures: 10
"""
    write_text(EDA_DIR / "eda_summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()
