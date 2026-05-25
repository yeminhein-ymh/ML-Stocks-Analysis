from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from app.core.analyzer import analyze_stock, prepare_stock, scan_stocks
from app.core.backtesting import backtest_portfolio, backtest_signal_strategy
from app.core.config import DEFAULT_WATCHLIST, RISK_RULES, SIGNAL_LOG_FILE, TRADE_LOG_FILE
from app.core.data import fetch_last_prices, market_data_healthcheck, normalize_tickers
from app.core.modeling import train_models
from app.core.paper_trading import cancel_order, log_signal, open_orders_table, place_paper_order, position_size
from app.core.screener import COMMON_UNIVERSE, screen_universe
from app.core.watchlists import load_watchlists, parse_uploaded_watchlist, save_watchlist

load_dotenv()

st.set_page_config(
    page_title="Universal AI Stock Analysis Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.35rem;}
        .risk-note {
            border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 14px;
            background: #f8fafc; color: #0f172a; font-size: 0.94rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(
        '<div class="risk-note"><b>Education and research only.</b> This dashboard is not financial advice, does not guarantee profit, and defaults to analysis plus Alpaca paper trading only. Live trading is intentionally not implemented.</div>',
        unsafe_allow_html=True,
    )


def ticker_selector(max_default: int = 10) -> list[str]:
    watchlists = load_watchlists()
    selected_watchlist = st.sidebar.selectbox("Watchlist", list(watchlists.keys()))
    manual = st.sidebar.text_input("Add any US ticker", placeholder="Example: AAPL, MSFT, AMD")
    uploaded = st.sidebar.file_uploader("Upload CSV watchlist", type=["csv"])
    tickers = list(watchlists[selected_watchlist])
    if uploaded is not None:
        tickers.extend(parse_uploaded_watchlist(uploaded))
    if manual:
        tickers.extend(manual.split(","))
    tickers = normalize_tickers(tickers)
    selected = st.sidebar.multiselect("Selected stocks", tickers, default=tickers[:max_default])
    if st.sidebar.button("Save selected as custom watchlist"):
        save_watchlist(f"Custom {pd.Timestamp.now().strftime('%Y-%m-%d %H%M')}", selected)
        st.sidebar.success("Watchlist saved.")
    return selected


def format_scan_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    numeric_cols = {
        "Price": "${:.2f}",
        "Daily Change %": "{:.2f}%",
        "RSI": "{:.1f}",
        "AI bullish probability": "{:.1f}%",
        "AI bearish probability": "{:.1f}%",
        "Buy zone": "${:.2f}",
        "Stop loss": "${:.2f}",
        "Take profit": "${:.2f}",
        "Risk-reward ratio": "{:.2f}",
        "Suitability score": "{:.0f}",
    }
    return df.style.format({k: v for k, v in numeric_cols.items() if k in df.columns})


def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP", line=dict(width=1.2, dash="dot")))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False)
    return fig


def page_market_overview(selected: list[str]) -> None:
    st.header("Market Overview")
    disclaimer()
    overview = fetch_last_prices(["SPY", "QQQ", "DIA", "IWM", *selected[:12]])
    if overview.empty:
        st.warning(market_data_healthcheck())
        st.caption("The dashboard is working, but live price data could not be downloaded.")
        return
    cols = st.columns(4)
    for idx, row in overview.head(4).iterrows():
        cols[idx % 4].metric(row["Ticker"], f"${row['Price']:.2f}", f"{row['Daily Change %']:.2f}%")
    st.dataframe(format_scan_table(overview), use_container_width=True, hide_index=True)


def page_multi_stock_scanner(selected: list[str], allow_penny: bool) -> pd.DataFrame:
    st.header("Multi-Stock AI Scanner")
    scan_size = st.select_slider("Scan size", options=[5, 10, 20, 50, 100], value=min(20, max(5, len(selected))))
    run = st.button("Run scanner", type="primary")
    if not run:
        st.info("Choose up to 100 stocks and run the scanner.")
        return pd.DataFrame()
    with st.spinner("Scanning selected stocks..."):
        table = scan_stocks(selected, limit=scan_size, allow_penny_stocks=allow_penny)
    if table.empty:
        st.warning("No scan results were available.")
        return table
    st.dataframe(format_scan_table(table), use_container_width=True, hide_index=True)
    st.download_button("Download scanner results", table.to_csv(index=False), "scanner_results.csv", "text/csv")
    return table


def page_individual_analysis(selected: list[str], allow_penny: bool) -> None:
    st.header("Individual Stock Analysis")
    ticker = st.selectbox("Ticker", selected or DEFAULT_WATCHLIST)
    with st.spinner(f"Analyzing {ticker}..."):
        result = analyze_stock(ticker, allow_penny_stocks=allow_penny)
    if "Error" in result:
        st.error(result["Error"])
        return
    suitability = result["Suitability detail"]
    ai = result["AI detail"]
    cols = st.columns(5)
    cols[0].metric("Price", f"${result['Price']:.2f}", f"{result['Daily Change %']:.2f}%")
    cols[1].metric("Suitability", result["Suitability"], f"{result['Suitability score']}/100")
    cols[2].metric("Bullish Probability", f"{ai['bullish_probability']:.1f}%")
    cols[3].metric("Bearish Probability", f"{ai['bearish_probability']:.1f}%")
    cols[4].metric("Final Signal", result["Final signal"])
    st.write("Suitability check:", result["Suitability reasons"])
    st.plotly_chart(price_chart(result["Data"].tail(252), ticker), use_container_width=True)


def page_technical_dashboard(selected: list[str]) -> None:
    st.header("Technical Indicator Dashboard")
    ticker = st.selectbox("Technical ticker", selected or DEFAULT_WATCHLIST)
    df = prepare_stock(ticker).dropna()
    if df.empty:
        st.warning("No technical data available.")
        return
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI", f"{latest['RSI']:.1f}")
    c2.metric("MACD", f"{latest['MACD']:.2f}", f"Signal {latest['MACD_Signal']:.2f}")
    c3.metric("ATR %", f"{latest['ATR_Pct']:.2f}%")
    c4.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig.add_hline(y=70, line_dash="dot")
    fig.add_hline(y=30, line_dash="dot")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.tail(60), use_container_width=True)


def page_ai_prediction(selected: list[str]) -> None:
    st.header("AI Prediction Model")
    ticker = st.selectbox("Model ticker", selected or DEFAULT_WATCHLIST)
    df = prepare_stock(ticker)
    with st.spinner("Training with time-series split..."):
        result = train_models(df)
    if "error" in result:
        st.warning(result["error"])
        return
    rows = [{"Model": name, "CV Accuracy": detail["cv_accuracy"]} for name, detail in result["models"].items()]
    st.dataframe(pd.DataFrame(rows).style.format({"CV Accuracy": "{:.2%}"}), use_container_width=True, hide_index=True)
    rf = result["models"].get("Random Forest", {}).get("model")
    if rf is not None:
        importance = pd.DataFrame({"Feature": result["features"], "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature").head(15))
    st.caption("Target: 5-day future return labels of bullish > 1.5%, bearish < -1.5%, otherwise neutral. Training uses time-series splits to reduce look-ahead bias.")


def page_backtesting(selected: list[str]) -> None:
    st.header("Backtesting")
    tickers = st.multiselect("Backtest stocks", selected or DEFAULT_WATCHLIST, default=(selected or DEFAULT_WATCHLIST)[:5])
    cost = st.number_input("Transaction cost", min_value=0.0, max_value=0.02, value=0.001, step=0.0005, format="%.4f")
    slippage = st.number_input("Slippage", min_value=0.0, max_value=0.02, value=0.001, step=0.0005, format="%.4f")
    if not st.button("Run backtest", type="primary"):
        return
    frames = {ticker: prepare_stock(ticker) for ticker in tickers}
    stats = []
    for ticker, df in frames.items():
        result = backtest_signal_strategy(df, cost, slippage)
        if "error" not in result:
            stats.append({"Ticker": ticker, **{k: v for k, v in result.items() if not k.endswith("_curve")}})
    if stats:
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
    portfolio = backtest_portfolio(frames, cost, slippage)
    if "error" not in portfolio:
        st.subheader("Portfolio Strategy")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", f"{portfolio['total_return']:.2%}")
        c2.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
        c3.metric("Maximum Drawdown", f"{portfolio['max_drawdown']:.2%}")
        st.line_chart(pd.DataFrame({"Equity": portfolio["equity_curve"], "Drawdown": portfolio["drawdown_curve"]}))


def page_paper_trading(selected: list[str], allow_penny: bool) -> None:
    st.header("Paper Trading Bot")
    st.warning("Paper trading only. Emergency stop disables scan/order actions in this session.")
    active = st.multiselect("Active bot stocks", selected or DEFAULT_WATCHLIST, default=(selected or DEFAULT_WATCHLIST)[:5])
    capital = st.number_input("Paper capital", min_value=1000.0, value=25000.0, step=1000.0)
    auto_scan = st.toggle("Auto-scan every 5 minutes", value=False)
    emergency_stop = st.toggle("Emergency stop", value=False)
    if auto_scan and st_autorefresh is not None:
        st_autorefresh(interval=5 * 60 * 1000, key="paper_bot_refresh")
    elif auto_scan:
        st.info("Install streamlit-autorefresh to enable timed browser refreshes.")
    if st.button("Scan and log paper signals", type="primary", disabled=emergency_stop):
        table = scan_stocks(active, limit=len(active), allow_penny_stocks=allow_penny)
        st.dataframe(format_scan_table(table), use_container_width=True, hide_index=True)
        for _, row in table.iterrows():
            log_signal(row.to_dict())
            if row["Final signal"] in {"Strong Buy", "Buy"}:
                qty = position_size(capital, row["Buy zone"], row["Stop loss"])
                st.write(f"{row['Ticker']}: eligible paper buy size = {qty} shares.")
    ticker = st.selectbox("Manual paper order ticker", active or DEFAULT_WATCHLIST)
    qty = st.number_input("Manual paper quantity", min_value=0, value=0, step=1)
    if st.button("Submit Alpaca paper buy", disabled=emergency_stop):
        result = place_paper_order(ticker, qty, side="buy")
        (st.success if result["ok"] else st.error)(result["message"])
    st.subheader("Open Paper Orders")
    open_orders = open_orders_table(ticker)
    if open_orders.empty:
        st.caption("No open paper orders for the selected ticker.")
    else:
        st.dataframe(open_orders, use_container_width=True, hide_index=True)
        cancel_id = st.selectbox("Cancel open order", open_orders["id"].tolist())
        if st.button("Cancel selected paper order", disabled=emergency_stop):
            result = cancel_order(cancel_id)
            (st.success if result["ok"] else st.error)(result["message"])
            st.rerun()
    if TRADE_LOG_FILE.exists():
        st.subheader("Trade Log")
        st.dataframe(pd.read_csv(TRADE_LOG_FILE).tail(100), use_container_width=True)
    if SIGNAL_LOG_FILE.exists():
        st.subheader("Signal Log")
        st.dataframe(pd.read_csv(SIGNAL_LOG_FILE).tail(100), use_container_width=True)


def page_risk_management() -> None:
    st.header("Risk Management")
    st.write("Built-in paper-bot constraints:")
    st.table(
        pd.DataFrame(
            [
                {"Rule": "Maximum risk per trade", "Value": f"{RISK_RULES['max_risk_per_trade']:.0%} of capital"},
                {"Rule": "Maximum exposure per stock", "Value": f"{RISK_RULES['max_exposure_per_stock']:.0%}"},
                {"Rule": "Maximum total exposure", "Value": f"{RISK_RULES['max_total_exposure']:.0%}"},
                {"Rule": "Minimum reward:risk", "Value": f"1:{RISK_RULES['min_reward_to_risk']:.0f}"},
            ]
        )
    )
    entry = st.number_input("Entry price", value=100.0)
    stop = st.number_input("Stop loss", value=95.0)
    capital = st.number_input("Capital", value=25000.0)
    st.metric("Position size", f"{position_size(capital, entry, stop)} shares")


def page_portfolio_allocation(selected: list[str], allow_penny: bool) -> None:
    st.header("Portfolio Allocation")
    if not st.button("Generate allocation from selected stocks"):
        st.info("Allocation uses suitable stocks with Buy or Strong Buy signals and caps each stock at 20%.")
        return
    table = scan_stocks(selected, limit=min(len(selected), 100), allow_penny_stocks=allow_penny)
    candidates = table[table["Final signal"].isin(["Strong Buy", "Buy"])].copy()
    if candidates.empty:
        st.warning("No allocation candidates passed the filters.")
        return
    candidates["Raw Weight"] = candidates["AI bullish probability"] * candidates["Suitability score"] * candidates["Risk-reward ratio"].clip(lower=1)
    candidates["Weight"] = candidates["Raw Weight"] / candidates["Raw Weight"].sum()
    candidates["Weight"] = candidates["Weight"].clip(upper=RISK_RULES["max_exposure_per_stock"])
    candidates["Weight"] = candidates["Weight"] / candidates["Weight"].sum() * min(1, RISK_RULES["max_total_exposure"])
    st.dataframe(candidates[["Ticker", "Final signal", "Weight", "AI bullish probability", "Risk-reward ratio"]].style.format({"Weight": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.bar_chart(candidates.set_index("Ticker")["Weight"])


def page_watchlist_manager(selected: list[str]) -> None:
    st.header("Watchlist Manager")
    st.write("Default watchlist")
    st.code(", ".join(DEFAULT_WATCHLIST))
    custom_name = st.text_input("Custom watchlist name", value="My Universal Watchlist")
    manual = st.text_area("Tickers", value=", ".join(selected or DEFAULT_WATCHLIST), height=120)
    if st.button("Save watchlist", type="primary"):
        save_watchlist(custom_name, manual.split(","))
        st.success("Saved custom watchlist.")
    st.subheader("Stock Screener")
    limit = st.select_slider("Screener universe size", options=[5, 10, 20, 50, 100], value=50)
    universe_text = st.text_area("Universe", value=", ".join(COMMON_UNIVERSE), height=100)
    if st.button("Run suitability screener"):
        result = screen_universe(normalize_tickers(universe_text.split(",")), limit=limit)
        st.dataframe(format_scan_table(result), use_container_width=True, hide_index=True)


def main() -> None:
    inject_style()
    st.title("Universal AI Stock Analysis Dashboard")
    selected = ticker_selector()
    allow_penny = st.sidebar.toggle("Manually allow penny stocks", value=False)
    page = st.sidebar.radio(
        "Dashboard pages",
        [
            "Market Overview",
            "Multi-Stock AI Scanner",
            "Individual Stock Analysis",
            "Technical Indicator Dashboard",
            "AI Prediction Model",
            "Backtesting",
            "Paper Trading Bot",
            "Risk Management",
            "Portfolio Allocation",
            "Watchlist Manager",
        ],
    )

    if not selected:
        selected = DEFAULT_WATCHLIST[:10]

    if page == "Market Overview":
        page_market_overview(selected)
    elif page == "Multi-Stock AI Scanner":
        page_multi_stock_scanner(selected, allow_penny)
    elif page == "Individual Stock Analysis":
        page_individual_analysis(selected, allow_penny)
    elif page == "Technical Indicator Dashboard":
        page_technical_dashboard(selected)
    elif page == "AI Prediction Model":
        page_ai_prediction(selected)
    elif page == "Backtesting":
        page_backtesting(selected)
    elif page == "Paper Trading Bot":
        page_paper_trading(selected, allow_penny)
    elif page == "Risk Management":
        page_risk_management()
    elif page == "Portfolio Allocation":
        page_portfolio_allocation(selected, allow_penny)
    elif page == "Watchlist Manager":
        page_watchlist_manager(selected)


if __name__ == "__main__":
    main()
