import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.notebook import tqdm

def backtest(model_predict, data, size=None, initial_capital=10_000_000,
             position_size=0.1):
    """
    Long-only backtest for Indonesian stocks (IDX).

    Args:
        model_predict : callable — returns predicted price for a dataset item
        data          : HuggingFace Dataset with prompt/completion + price fields
        size          : number of items to evaluate (default: all)
        initial_capital: starting capital in IDR
        position_size : fraction of capital risked per trade (default 10%)
    """
    TRANSACTION_COST = 0.004   # 0.15% buy + 0.25% sell = 0.4% round trip (IDX)
    SLIPPAGE         = 0.002   # 0.2% for low-liquidity IDX small caps
    STOP_LOSS = -0.03

    if size is None:
        size = len(data)

    capital = initial_capital
    trades  = []

    for i in tqdm(range(size)):
        item = data[i]

        # Support both datasets: with explicit fields OR prompt-only
        if "Last Price" in item:
            last_price   = float(item["Last Price"])
            future_price = float(item["Future Price"])
            actual_pct   = float(item["Return %"])
            ticker       = item.get("Ticker", "N/A")
            entry_date   = item.get("End Date", "")
            exit_date    = item.get("Future Date", "")

        # Model prediction → direction
        predicted_price = model_predict(item)
        predicted_pct   = (predicted_price - last_price) / last_price * 100

        if predicted_pct > 0:   # UP → enter long with fixed position size
            actual_return = actual_pct / 100
            actual_return = max(actual_return, STOP_LOSS)
            position   = capital * position_size
            net_return = actual_return - TRANSACTION_COST - SLIPPAGE
            capital    = capital + (position * net_return)  # only position at risk
            entered    = True
        else:                   # DOWN → hold cash
            net_return = 0
            entered    = False

        trades.append({
            "Ticker":        ticker,
            "Entry Date":    entry_date,
            "Exit Date":     exit_date,
            "Last Price":    last_price,
            "Future Price":  future_price,
            "Predicted Pct": predicted_pct,
            "Actual Pct":    actual_pct,
            "Net Return %":  net_return * 100,
            "Entered":       entered,
            "Capital":       capital,
        })

    return pd.DataFrame(trades)

def backtest_report(trades, ticker, initial_capital=10_000_000, position_size=0.1):
    entered = trades[trades["Entered"]]

    total_return  = (trades["Capital"].iloc[-1] / initial_capital - 1) * 100
    win_rate      = (entered["Net Return %"] > 0).mean() * 100 if len(entered) > 0 else 0
    avg_win       = entered[entered["Net Return %"] > 0]["Net Return %"].mean() if len(entered) > 0 else 0
    avg_loss      = entered[entered["Net Return %"] < 0]["Net Return %"].mean() if len(entered) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    max_drawdown  = (trades["Capital"] / trades["Capital"].cummax() - 1).min() * 100
    n_trades      = len(entered)
    n_skipped     = len(trades) - n_trades

    # -----------------------------
    # TRUE BUY & HOLD
    # -----------------------------
    first_price = test_raw[0]["Last Price"]
    last_price  = trades["Future Price"].iloc[-1]

    shares = initial_capital / first_price

    bah_final = shares * last_price

    bah_return = (
        bah_final / initial_capital - 1
    ) * 100
    alpha      = total_return - bah_return

    # -----------------------------
    # IHSG / JKSE BENCHMARK
    # -----------------------------
    ihsg = yf.Ticker("^JKSE").history(period="max")

    ihsg.index = ihsg.index.tz_localize(None)

    trade_dates = pd.to_datetime(trades["Exit Date"])

    ihsg = ihsg.loc[
        (ihsg.index >= trade_dates.min()) &
        (ihsg.index <= trade_dates.max())
    ]

    # Align with trade dates
    ihsg_close = ihsg["Close"].reindex(
        trade_dates,
        method="ffill"
    )

    # Normalize to initial capital
    ihsg_curve = (
        ihsg_close / ihsg_close.iloc[0]
    ) * initial_capital
    ihsg_return = (
        ihsg_curve.iloc[-1] / ihsg_curve.iloc[0] - 1
    ) * 100

    print(f"\n{'='*48}")
    print(f"  Backtest Results (Long-Only, IDX)")
    print(f"  Position Size: {position_size*100:.0f}% per trade")
    print(f"{'='*48}")
    print(f"  Total Return     : {total_return:.1f}%")
    print(f"  Buy & Hold       : {bah_return:.1f}%")
    print(f"  IHSG Return      : {ihsg_return:.1f}%")
    print(f"  Alpha            : {alpha:+.1f}%")
    print(f"  Win Rate         : {win_rate:.1f}%")
    print(f"  Avg Win          : {avg_win:.2f}%")
    print(f"  Avg Loss         : {avg_loss:.2f}%")
    print(f"  Profit Factor    : {profit_factor:.2f}")
    print(f"  Max Drawdown     : {max_drawdown:.1f}%")
    print(f"  Trades Entered   : {n_trades}")
    print(f"  Trades Skipped   : {n_skipped}")
    print(f"{'='*48}\n")

    bah_curve = (
        trades["Future Price"] / first_price
    ) * initial_capital

    # -----------------------------
    # PLOT
    # -----------------------------
    fig = go.Figure()

    # Strategy
    fig.add_trace(go.Scatter(
        x=trade_dates,
        y=trades["Capital"],
        mode="lines",
        name="NOVA AI",
        line=dict(color="blue", width=2)
    ))

    # Buy & Hold
    fig.add_trace(go.Scatter(
        x=trade_dates,
        y=bah_curve[1:],
        mode="lines",
        name="Buy & Hold",
        line=dict(color="red", width=2)
    ))

    # IHSG Benchmark
    fig.add_trace(go.Scatter(
        x=trade_dates,
        y=ihsg_curve,
        mode="lines",
        name="IHSG",
        line=dict(color="purple", width=2)
    ))

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Modal Awal"
    )

    fig.update_layout(
        title=f"{ticker}",
        xaxis_title="Tanggal",
        yaxis_title="Nilai Investasi (IDR)",
        width=1400,
        height=500,
        template="plotly_white",
        showlegend=True,
    )

    fig.show()

    return trades