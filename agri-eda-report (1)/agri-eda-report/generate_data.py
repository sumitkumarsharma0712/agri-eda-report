"""Generate hypothetical agribusiness data for the EDA demonstration.

Creates two files (all values are synthetic):
  data/farm_level.csv      - one row per farm-season (Vidarbha-style survey)
  data/monthly_prices.csv  - monthly modal mandi price of soybean, 2021-2025
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------- farm-level data ----------------
N = 900
districts = ["Nagpur", "Wardha", "Amravati", "Yavatmal", "Akola"]
dist_rain = {"Nagpur": 1000, "Wardha": 950, "Amravati": 850, "Yavatmal": 900, "Akola": 780}
dist_irrig = {"Nagpur": 0.45, "Wardha": 0.35, "Amravati": 0.30, "Yavatmal": 0.22, "Akola": 0.25}
crops = ["soybean", "cotton", "wheat", "tur"]
base = {"soybean": 10.5, "cotton": 9.5, "wheat": 26.0, "tur": 7.5}       # q/ha
price = {"soybean": 4600, "cotton": 7000, "wheat": 2300, "tur": 7200}     # Rs/quintal
cost = {"soybean": 32000, "cotton": 52000, "wheat": 36000, "tur": 30000}  # Rs/ha

rows = []
for i in range(N):
    d = rng.choice(districts)
    c = rng.choice(crops, p=[0.35, 0.30, 0.20, 0.15])
    rain = rng.normal(dist_rain[d], 130)
    irr = int(rng.random() < dist_irrig[d])
    fert = max(10, rng.normal(85, 22))
    area = round(float(rng.gamma(2.0, 0.9) + 0.3), 2)
    y = base[c] * (1 + 0.0004 * (rain - 850) + 0.0035 * (fert - 85) + 0.18 * irr) + rng.normal(0, 1.4)
    y = max(1.5, y)
    p = price[c] * rng.normal(1, 0.06)
    rows.append({
        "farm_id": f"F{1000 + i}", "district": d, "crop": c,
        "season": "Rabi" if c == "wheat" else "Kharif",
        "area_ha": area, "rainfall_mm": round(rain), "irrigated": irr,
        "fertiliser_kg_ha": round(fert), "yield_q_ha": round(y, 1),
        "price_rs_q": round(p), "cost_rs_ha": round(cost[c] * rng.normal(1, 0.08)),
    })
df = pd.DataFrame(rows)
df["revenue_rs_ha"] = (df["yield_q_ha"] * df["price_rs_q"]).round()
df["profit_rs_ha"] = df["revenue_rs_ha"] - df["cost_rs_ha"]

# inject a few anomalies and gaps (to be found by the EDA)
bad = rng.choice(df.index, 6, replace=False)
df.loc[bad, "yield_q_ha"] = df.loc[bad, "yield_q_ha"] * 4
df.loc[rng.choice(df.index, 40, replace=False), "fertiliser_kg_ha"] = np.nan
df.loc[rng.choice(df.index, 25, replace=False), "rainfall_mm"] = np.nan
df.to_csv("data/farm_level.csv", index=False)

# ---------------- monthly price series ----------------
idx = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
t = np.arange(len(idx))
seasonal = {1: 1.02, 2: 1.04, 3: 1.06, 4: 1.08, 5: 1.10, 6: 1.11, 7: 1.09,
            8: 1.05, 9: 1.00, 10: 0.92, 11: 0.90, 12: 0.96}   # harvest dip Oct-Nov
s = np.array([seasonal[m.month] for m in idx])
p = (4000 + 18 * t) * s * rng.normal(1, 0.015, len(idx))
p[idx == "2023-06-01"] *= 1.28      # supply-shock spike
p[idx == "2024-11-01"] *= 0.82      # glut / data issue
mp = pd.DataFrame({"month": idx, "soy_price_rs_q": p.round()})
mp.to_csv("data/monthly_prices.csv", index=False)
print(len(df), "farm rows;", len(mp), "monthly price rows")
