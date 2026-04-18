#!/usr/bin/env python3
"""
SwingScan AI Signal Scorer — Training Script
============================================
Run this locally whenever you have new backtest CSV data.
Outputs: signal_model.pkl  (deploy this alongside api_server.py)

Usage:
    python train_model.py backtest1.csv backtest2.csv ...
    python train_model.py              # uses all CSV files in current dir
"""

import sys, glob, json, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def build_features(df):
    df = df.copy()
    df["is_sell"]    = (df["signal"] == "SELL").astype(int)
    df["is_hour_12"] = (df["hour"] == 12).astype(int)
    df["is_hour_11"] = (df["hour"] == 11).astype(int)
    df["is_hour_13"] = (df["hour"] == 13).astype(int)
    df["is_hour_10"] = (df["hour"] == 10).astype(int)
    df["adx_strong"] = (df["adx"] >= 40).astype(int)
    df["adx_very"]   = (df["adx"] >= 55).astype(int)
    df["score_high"] = (df["score"] >= 11).astype(int)
    df["rsi_zone"]   = pd.cut(df["rsi"], bins=[0,35,50,65,100],
                              labels=[0,1,2,3]).astype(int)
    df["sl_pct"]     = abs(df["entry"] - df["stop_loss"]) / df["entry"] * 100
    df["tgt_pct"]    = abs(df["target"] - df["entry"])    / df["entry"] * 100
    return df

FEATURES = [
    "score", "adx", "rsi", "hour",
    "is_sell", "is_hour_12", "is_hour_11", "is_hour_13", "is_hour_10",
    "adx_strong", "adx_very", "score_high", "rsi_zone",
    "sl_pct", "tgt_pct",
]

def main():
    files = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("backtest_*.csv")
    if not files:
        print("No CSV files found. Pass filenames as arguments.")
        return

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} trades")

    df = pd.concat(dfs, ignore_index=True)
    df.drop_duplicates(subset=["symbol", "time", "signal"], inplace=True)
    df = build_features(df)
    print(f"\nTotal unique trades: {len(df)}")

    X = df[FEATURES].values
    y = (df["outcome"] == "WIN").astype(int).values
    print(f"Win rate: {y.mean()*100:.1f}%")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    )

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"Cross-val AUC: {auc:.3f}")

    model.fit(X, y)

    payload = {
        "model":       model,
        "features":    FEATURES,
        "baseline_wr": float(y.mean()),
        "auc":         float(auc),
        "n_trades":    len(X),
    }
    with open("signal_model.pkl", "wb") as f:
        pickle.dump(payload, f)

    with open("signal_model_meta.json", "w") as f:
        json.dump({k: v for k, v in payload.items() if k != "model"}, f, indent=2)

    print("\nModel saved → signal_model.pkl")
    print("Metadata   → signal_model_meta.json")
    print("\nDeploy signal_model.pkl alongside api_server.py on Render.")
    print("\nWin rate by confidence threshold:")
    probs = model.predict_proba(X)[:,1]
    for t in [0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = probs >= t
        if mask.sum() == 0: continue
        wr = y[mask].mean()
        print(f"  ≥{t:.0%}: {mask.sum():3d} trades ({mask.mean()*100:.0f}% kept) "
              f"| WR {wr*100:.1f}%")

if __name__ == "__main__":
    main()
