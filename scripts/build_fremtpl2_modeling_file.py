"""Build the merged freMTPL2 policy-claim modeling CSV."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fremtpl2_utils import (
    MODELING_FILE,
    OUTPUT_DIR,
    build_modeling_frame,
    ensure_directories,
    read_freq,
    read_sev,
    weighted_claim_rate,
    weighted_pure_premium,
    write_text,
)


def main() -> None:
    ensure_directories()
    freq = read_freq()
    sev = read_sev()
    df = build_modeling_frame(freq, sev)

    MODELING_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODELING_FILE, index=False)

    summary = f"""freMTPL2 modeling file summary

Raw frequency rows: {len(freq):,}
Raw severity claim rows: {len(sev):,}
Final modeling rows: {len(df):,}
Final columns: {len(df.columns):,}
Unique policies: {df["IDpol"].nunique():,}
Policies with recorded ClaimNb > 0: {(df["ClaimNb"] > 0).sum():,}
Policies with joined claim amount > 0: {(df["TotalClaimAmount"] > 0).sum():,}
Rows with ClaimNb / severity-row mismatch: {df["ClaimCountMismatch"].sum():,}
Total exposure: {df["Exposure"].sum():,.2f}
Total claim count: {df["ClaimNb"].sum():,.0f}
Total claim amount: {df["TotalClaimAmount"].sum():,.2f}
Portfolio claim frequency: {weighted_claim_rate(df):.6f}
Portfolio pure premium: {weighted_pure_premium(df):.4f}

Output CSV:
{MODELING_FILE}
"""
    write_text(OUTPUT_DIR / "modeling" / "data_build_summary.txt", summary)
    print(summary)


if __name__ == "__main__":
    main()
