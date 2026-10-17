"""Exploratory Data Analysis for the hypothetical agribusiness dataset.

Run:  python eda_analysis.py
Outputs: figures/*.png and reports/*.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})

df = pd.read_csv("data/farm_level.csv")
mp = pd.read_csv("data/monthly_prices.csv", parse_dates=["month"])


def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=170, facecolor="white")
    plt.close()


# ---------- 1. data profile ----------
prof = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_pct": (df.isna().mean() * 100).round(1),
                     "n_unique": df.nunique()})
prof.to_csv("reports/profile.csv")
df.describe().T.round(2).to_csv("reports/summary_stats.csv")

# ---------- 2. histogram: yield distribution (soybean) ----------
s = df.loc[df.crop == "soybean", "yield_q_ha"]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.hist(s[s < 30], bins=24, color=BLUE, edgecolor="white", linewidth=1.2)
ax.axvline(s.median(), color=INK, lw=1.2, ls="--")
ax.text(s.median() + 0.3, ax.get_ylim()[1] * 0.92, f"median {s.median():.1f} q/ha", color=INK, fontsize=9)
ax.set_title("Soybean yield distribution (farm-level)")
ax.set_xlabel("Yield (quintal per hectare)"); ax.set_ylabel("Number of farms")
save("fig1_histogram")

# ---------- 3. bar: mean yield by district ----------
clean = df[df.yield_q_ha < df.groupby("crop").yield_q_ha.transform(lambda x: x.quantile(.99))]
sb = clean[clean.crop == "soybean"].groupby("district").yield_q_ha.mean().sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.4))
bars = ax.barh(sb.index, sb.values, color=BLUE, height=0.6)
for b, v in zip(bars, sb.values):
    ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", color=INK, fontsize=9)
ax.set_xlim(0, sb.max() * 1.15)
ax.grid(axis="y", visible=False)
ax.set_title("Average soybean yield by district")
ax.set_xlabel("Yield (quintal per hectare)")
save("fig2_bar_district")

# ---------- 4. line: monthly price with anomalies ----------
mp = mp.set_index("month")
roll = mp.soy_price_rs_q.rolling(12, center=True, min_periods=6).median()
resid = mp.soy_price_rs_q - roll
z = (resid - resid.median()) / (1.4826 * (resid - resid.median()).abs().median())
anom = mp[z.abs() > 3.5]
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.plot(mp.index, mp.soy_price_rs_q, color=BLUE, lw=2, label="Monthly modal price")
ax.plot(roll.index, roll, color=MUTED, lw=1.4, ls="--", label="12-month rolling median (trend)")
ax.scatter(anom.index, anom.soy_price_rs_q, s=60, color=ORANGE, zorder=5, edgecolor="white", linewidth=1.5, label="Flagged anomaly")
ax.set_title("Soybean modal price, monthly (hypothetical)")
ax.set_ylabel("Rs per quintal"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
save("fig3_price_trend")
anom.assign(robust_z=z[anom.index].round(1)).to_csv("reports/price_anomalies.csv")

# ---------- 5. seasonal index ----------
si = (mp.soy_price_rs_q / roll).groupby(mp.index.month).median()
fig, ax = plt.subplots(figsize=(6.2, 3.3))
cols = [ORANGE if v < 0.95 else BLUE for v in si.values]
ax.bar(si.index, si.values, color=cols, width=0.7)
ax.axhline(1, color=INK, lw=1)
ax.set_ylim(0.8, 1.2); ax.set_xticks(range(1, 13))
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.grid(axis="x", visible=False)
ax.set_title("Seasonal price index (1.00 = trend)")
ax.set_xlabel("Month"); ax.set_ylabel("Index")
save("fig4_seasonal_index")
si.round(3).to_csv("reports/seasonal_index.csv", header=["index"])

# ---------- 6. scatter: rainfall vs yield by irrigation ----------
sc = clean[(clean.crop == "soybean")].dropna(subset=["rainfall_mm"])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for v, col, lab in [(0, BLUE, "Rain-fed"), (1, ORANGE, "Irrigated")]:
    d = sc[sc.irrigated == v]
    ax.scatter(d.rainfall_mm, d.yield_q_ha, s=26, color=col, alpha=.75, edgecolor="white", linewidth=.6, label=lab)
    m, b = np.polyfit(d.rainfall_mm, d.yield_q_ha, 1)
    xs = np.linspace(d.rainfall_mm.min(), d.rainfall_mm.max(), 50)
    ax.plot(xs, m * xs + b, color=col, lw=2)
ax.set_title("Rainfall vs soybean yield")
ax.set_xlabel("Seasonal rainfall (mm)"); ax.set_ylabel("Yield (q/ha)")
ax.legend(frameon=False, fontsize=9)
save("fig5_scatter")

# ---------- 7. box plot: yield by crop (outlier view) ----------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
order = ["soybean", "cotton", "tur", "wheat"]
data = [df.loc[df.crop == c, "yield_q_ha"].dropna() for c in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55,
                flierprops=dict(marker="o", markerfacecolor=ORANGE, markeredgecolor="white", markersize=6),
                medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
for b in bp["boxes"]:
    b.set_facecolor(BLUE); b.set_alpha(.85); b.set_edgecolor("white")
ax.set_title("Yield by crop; points beyond whiskers flagged")
ax.set_ylabel("Yield (q/ha)"); ax.grid(axis="x", visible=False)
save("fig6_boxplot")

# ---------- 8. correlation heatmap ----------
num = clean[["yield_q_ha", "rainfall_mm", "fertiliser_kg_ha", "irrigated", "area_ha", "cost_rs_ha", "profit_rs_ha"]]
corr = num.corr(method="spearman")
corr.round(2).to_csv("reports/correlation_spearman.csv")
fig, ax = plt.subplots(figsize=(6.2, 4.9))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
lbl = ["yield", "rainfall", "fertiliser", "irrigated", "area", "cost", "profit"]
ax.set_xticklabels(lbl, rotation=40, ha="right"); ax.set_yticklabels(lbl)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > .55 else INK)
ax.grid(False)
ax.set_title("Spearman correlation matrix")
fig.colorbar(im, ax=ax, shrink=.8)
save("fig7_correlation")

# ---------- 9. profitability by crop (grouped summary) ----------
prof_crop = clean.groupby("crop").agg(farms=("farm_id", "count"), yield_q_ha=("yield_q_ha", "mean"),
                                     price=("price_rs_q", "mean"), cost=("cost_rs_ha", "mean"),
                                     profit=("profit_rs_ha", "mean")).round(0)
prof_crop.to_csv("reports/profit_by_crop.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.4))
pc = prof_crop.sort_values("profit")
bars = ax.barh(pc.index, pc.profit, color=[ORANGE if v < 0 else BLUE for v in pc.profit], height=.6)
for b, v in zip(bars, pc.profit):
    ax.text(v + (800 if v >= 0 else -800), b.get_y() + b.get_height() / 2, f"Rs {v:,.0f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=1); ax.grid(axis="y", visible=False)
ax.set_xlim(min(0, pc.profit.min()) * 1.5 - 5000, pc.profit.max() * 1.3)
ax.set_title("Average profit per hectare by crop")
ax.set_xlabel("Rupees per hectare")
save("fig8_profit_crop")

# ---------- 10. anomaly report (IQR within crop) ----------
g = df.groupby("crop").yield_q_ha
q1, q3 = g.transform(lambda x: x.quantile(.25)), g.transform(lambda x: x.quantile(.75))
iqr = q3 - q1
flag = df[(df.yield_q_ha < q1 - 1.5 * iqr) | (df.yield_q_ha > q3 + 1.5 * iqr)]
flag.to_csv("reports/yield_outliers.csv", index=False)

# console summary used in the report
print("Rows:", len(df), "| outliers flagged:", len(flag))
print("Price anomalies:\n", anom.assign(z=z[anom.index].round(1)))
print("Seasonal index min month:", si.idxmin(), round(si.min(), 3), "| max month:", si.idxmax(), round(si.max(), 3))
print(corr.round(2).to_string())
print(prof_crop.to_string())
print(sb.round(2).to_string())
print(prof.to_string())
