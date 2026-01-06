import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Black-Scholes with dividend yield q
# -----------------------------
@dataclass
class OptionInputs:
    s: float
    k: float
    t: float
    r: float
    q: float
    sigma: float

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1_d2(inp: OptionInputs) -> Tuple[float, float]:
    if inp.t <= 0 or inp.s <= 0 or inp.k <= 0 or inp.sigma <= 0:
        return float("nan"), float("nan")
    vol_sqrt_t = inp.sigma * math.sqrt(inp.t)
    d1 = (math.log(inp.s / inp.k) + (inp.r - inp.q + 0.5 * inp.sigma * inp.sigma) * inp.t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2

def bs_price(inp: OptionInputs, option_type: str) -> float:
    d1, d2 = d1_d2(inp)
    if not np.isfinite(d1) or not np.isfinite(d2):
        return float("nan")
    disc_r = math.exp(-inp.r * inp.t)
    disc_q = math.exp(-inp.q * inp.t)

    if option_type == "call":
        return float(inp.s * disc_q * norm_cdf(d1) - inp.k * disc_r * norm_cdf(d2))
    return float(inp.k * disc_r * norm_cdf(-d2) - inp.s * disc_q * norm_cdf(-d1))

def implied_vol_bisect(
    target_price: float,
    option_type: str,
    base: OptionInputs,
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    max_iter: int = 120,
    tol: float = 1e-6,
) -> Optional[float]:
    if target_price <= 0 or base.t <= 0 or base.s <= 0 or base.k <= 0:
        return None
    if option_type not in ("call", "put"):
        return None

    def price_for_sigma(sig: float) -> float:
        inp = OptionInputs(base.s, base.k, base.t, base.r, base.q, sig)
        return bs_price(inp, option_type)

    lo, hi = sigma_low, sigma_high
    plo, phi = price_for_sigma(lo), price_for_sigma(hi)

    if not np.isfinite(plo) or not np.isfinite(phi):
        return None
    if (plo - target_price) * (phi - target_price) > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmid = price_for_sigma(mid)
        if not np.isfinite(pmid):
            return None
        err = pmid - target_price
        if abs(err) < tol:
            return float(mid)
        if (plo - target_price) * err <= 0:
            hi = mid
            phi = pmid
        else:
            lo = mid
            plo = pmid

    return float(0.5 * (lo + hi))


# -----------------------------
# Helpers for user input
# -----------------------------
def parse_lines_to_table(text: str) -> pd.DataFrame:
    """
    Accepts lines like:
    80, 22.1
    90 14.0
    100; 8.2
    Ignores empty lines and comments starting with #.
    """
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.replace(";", ",").replace("\t", ",").replace(" ", ",")
        parts = [p for p in line.split(",") if p != ""]
        if len(parts) < 2:
            continue
        try:
            k = float(parts[0])
            p = float(parts[1])
            rows.append((k, p))
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["strike", "market_price"])

def example_text() -> str:
    return "\n".join([
        "# strike, market_price",
        "80, 22",
        "90, 14",
        "100, 8",
        "110, 4",
        "120, 2",
    ])


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="IV Smile (Manual)", layout="wide")
st.title("IV Smile (Manual)")
st.caption("Paste strike and market price pairs. Get implied vols, a smile chart, and a CSV export.")

tabs = st.tabs(["Calculator", "About the Author"])

with tabs[0]:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1) Market inputs")

        option_side = st.selectbox("Option side", ["call", "put"], index=0)

        s = st.number_input("Spot S", min_value=0.01, value=100.0, step=0.5)
        t_days = st.number_input("Maturity (days)", min_value=1, value=60, step=1)

        r_pct = st.number_input("Risk-free rate r (%)", value=3.0, step=0.25)
        q_pct = st.number_input("Dividend yield q (%)", value=0.0, step=0.25)

        r = float(r_pct) / 100.0
        q = float(q_pct) / 100.0
        T = float(t_days) / 365.0

        st.subheader("2) Paste strikes and prices")
        st.write("One pair per line. You can use comma, space, or semicolon.")
        text = st.text_area("Strike and market price", value=example_text(), height=180)

        run = st.button("Compute implied vols", type="primary")

    with right:
        st.subheader("Results")

        if not run:
            st.info("Click Compute implied vols.")
            st.stop()

        df = parse_lines_to_table(text)

        if df.empty:
            st.error("No valid rows found. Paste lines like: 100, 8.5")
            st.stop()

        df = df.dropna().copy()
        df = df[(df["strike"] > 0) & (df["market_price"] > 0)].copy()

        if len(df) < 3:
            st.error("Add at least 3 valid points to build a smile.")
            st.stop()

        base = OptionInputs(s=float(s), k=0.0, t=float(T), r=float(r), q=float(q), sigma=0.2)

        ivs = []
        for _, row in df.iterrows():
            k = float(row["strike"])
            mp = float(row["market_price"])
            base_k = OptionInputs(base.s, k, base.t, base.r, base.q, base.sigma)
            iv = implied_vol_bisect(mp, option_side, base_k)
            ivs.append(iv if iv is not None else np.nan)

        out = df.copy()
        out["iv"] = np.array(ivs, dtype=float)
        out = out.dropna(subset=["iv"]).sort_values("strike").reset_index(drop=True)

        if out.empty:
            st.error("Could not compute implied vols. Check prices, maturity, and option side.")
            st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("Points used", f"{len(out)}")
        col2.metric("ATM strike (closest)", f"{out.iloc[(out['strike']-s).abs().argsort()[:1]]['strike'].iloc[0]:.2f}")
        col3.metric("Avg IV", f"{out['iv'].mean()*100:.2f}%")

        fig = plt.figure()
        plt.plot(out["strike"].values, out["iv"].values)
        plt.xlabel("Strike")
        plt.ylabel("Implied volatility")
        plt.title(f"IV Smile ({option_side})")
        st.pyplot(fig, clear_figure=True)

        st.dataframe(out, use_container_width=True)

        st.download_button(
            "Download CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="iv_smile_manual.csv",
            mime="text/csv",
        )

        st.subheader("Quick sanity checks")
        st.write("If IV looks wrong:")
        st.write("1) Check option side (call vs put).")
        st.write("2) Increase maturity days if prices are high.")
        st.write("3) Use realistic prices: deep ITM options cost more than OTM.")

tabs = st.tabs(["Application", "About the Author"])

with tabs[0]:
    st.subheader("Application")
    st.write("Main content of the app goes here.")

with tabs[1]:
    st.subheader("About the Author")
    st.write("Author: Maxime Dralez")
    st.write("LinkedIn:")
    st.link_button(
        "Open LinkedIn profile",
        "https://www.linkedin.com/in/maxime-dralez-finance"
    )
    st.write("")
    st.write("This app is a small educational tool.")
    st.write(
        "It turns a list of market quotes into an implied volatility smile you can export."
    )
