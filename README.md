# Universal AI Stock Analysis Dashboard

Streamlit dashboard and paper-trading bot for universal multi-stock analysis across liquid US stocks. It is no longer limited to the original fixed ticker set.

## Features

- Manual entry for any US ticker.
- Multi-select scanning for 5, 10, 20, 50, or 100 symbols.
- Default watchlist: NVDA, AMZN, META, GOOGL, PLTR, MU, SNDK, NBIS, AAPL, MSFT, TSLA, AMD, AVGO, SMCI, NFLX, ARM, TSM, ORCL, CRM, QCOM, INTC, COIN, MSTR, SOFI, HOOD, JPM, BAC, XOM, NEE, ENPH, FSLR.
- CSV watchlist upload and saved custom watchlists.
- Stock screener for formula suitability.
- Signal Synergy filters on the Multi-Stock AI Scanner: High Conviction Only, Buy-below-70% AI warning, and AI conviction highlighting.
- Daily signal journal for scanner and technical-indicator snapshots.
- Daily, weekly, and monthly Technical Indicator Dashboard snapshot journal for every selected stock.
- Sector divergence view in the Technical Snapshot Journal comparing AMAT/SNDK/INTC against AAPL/ABBV.
- Daily Multi-Stock AI Scanner reference archive with one-month through one-year history views.
- Automatic daily Top 50 autosave for scanner results, technical snapshots, signal journal rows, and Excel signal reports.
- Signal accuracy analysis by final signal after 1, 5, 10, or 20 trading days, with Strong Buy vs Synergy Strong Buy comparison and Hold support validation.
- Options AI Scanner for paper-only single-leg call/put candidates using Alpaca option contracts.
- Short Strategy Stock Scanner for educational low-float short setup research: Gap Up Short, Bounce Short, First Red Day, and Watchlist Monitor.
- Options Paper Trading page for paper limit orders on selected or manually entered option contracts.
- Suitability gate before AI signals.
- Individual Stock Analysis includes EMA 20, EMA 60, EMA 250, Daily Session VWAP, Anchored VWAP, and renamed Cumulative VWAP.
- Random Forest and optional XGBoost training with time-series splits.
- Individual and portfolio backtesting with cost and slippage.
- ATR-based Risk Management calculator with a default 2.0 reward:risk lock.
- Alpaca paper-trading helpers only. No live trading implementation.

## Pages

1. Market Overview
2. Multi-Stock AI Scanner
3. Short Strategy Stock Scanner
4. Options AI Scanner
5. Individual Stock Analysis
6. Technical Indicator Dashboard
7. AI Prediction Model
8. Backtesting
9. Signal Accuracy Analysis
10. Paper Trading Bot
11. Options Paper Trading
12. Risk Management
13. Portfolio Allocation
14. Watchlist Manager

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

For this machine, if you prefer the installed Python path already used in prior work:

```powershell
& 'C:\Users\Ye Min Hein\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run streamlit_app.py
```

## CSV Watchlist Format

The upload accepts a column named `ticker`, `symbol`, `tickers`, or `symbols`. If none exists, the first column is used.

```csv
ticker
AAPL
MSFT
AMD
```

## Trading Safety

This project is for education and research, not financial advice. It does not guarantee profits. Default behavior is analysis and Alpaca paper trading only. Live trading is intentionally not implemented.

Paper bot risk controls:

- Maximum risk per trade: 2% of capital
- Maximum exposure per stock: 20%
- Maximum total exposure: 80%
- Minimum reward:risk ratio: 1:2
- Emergency stop disables scan/order actions in the current session

## Technical Indicator Records

On the Technical Indicator Dashboard page, choose the stocks selected in the sidebar and record Daily, Weekly, and Monthly snapshots. Records are saved to:

```text
logs/technical_indicator_snapshots.csv
```

Each record includes ticker, period, period start/end, price, period return, high, low, volume, RSI, MACD, VWAP status, trend, ATR %, volume ratio, AI probabilities, suitability, and final signal. Re-recording the same ticker and period on the same date replaces that row to avoid duplicates.

Daily technical snapshots are also included in Signal Accuracy Analysis as the source `Daily Technical Snapshot Journal`, so the Accuracy by Source table reflects all selected technical-dashboard stocks instead of only the manually recorded ticker.

The Technical Indicator Dashboard also includes a Daily Calendar Archive target from June 26, 2026 through December 31, 2026. Use `Backfill missing calendar dates from 2026-06-26 to today` to create rows for every calendar date, including weekends. Weekend and market-closed rows use the latest available market bar and are marked in the `Market status` column.

## Recorded Signal Journal

The Signal Accuracy Analysis page includes a Daily Signal Journal Recorder. It records today's selected stock signals into:

```text
logs/daily_signal_analysis.csv
```

Use the journal range control to review Last 1 month, Last 3 months, Last 6 months, Last 1 year, Last 2 years, Last 5 years, All history, or a custom date range. Accuracy summaries can be grouped daily, monthly, or yearly. Re-recording the same ticker/source on the same date replaces that row, while older dates are kept for long-term reference.

The app keeps a rolling 365-day local archive and writes backup CSVs under:

```text
logs/archive_backups/
```

If running on Streamlit Cloud, files under `logs/` can reset after app redeploy/reboot because Streamlit Cloud does not provide permanent database storage. Download the full signal journal regularly, or use the Signal Accuracy Analysis import tool to restore a previous CSV/Excel journal into the current app storage.

The Signal Accuracy Analysis page also includes automatic daily Top 50 autosave. When enabled, it saves:

```text
logs/daily_signal_analysis.csv
logs/multi_stock_ai_scanner_history.csv
logs/technical_indicator_snapshots.csv
logs/daily_excel_reports/daily_top50_signal_analysis_YYYY-MM-DD.xlsx
logs/daily_excel_reports/signal_accuracy_1d_YYYY-MM-DD.xlsx
```

Excel reports include an `All Signals` sheet plus separate sheets for Strong Buy, Buy, Hold, Sell, and Avoid. The default evaluation holding period is 1 day, with selectable periods for 3 days, 7 days, 2 weeks, and 1 month.

## Multi-Stock Scanner Records

The Multi-Stock AI Scanner page can save each full scanner table into:

```text
logs/multi_stock_ai_scanner_history.csv
```

Keep `Save full scanner table to daily one-year reference archive` enabled when running the scanner. The page shows the saved reference archive with Last 1 month, Last 3 months, Last 6 months, Last 1 year, Last 2 years, Last 5 years, and All history filters. Re-running the same ticker on the same date replaces that row.

## Short Strategy Stock Scanner

The Short Strategy Stock Scanner is an educational scanner for low-float/small-cap short setup research. It applies global filters for minimum price, maximum market cap, excluded sectors, and excluded countries, then provides four tabs:

- Gap Up Short: identifies large gap-up runners, post-open push, consolidation high/low, crack alerts, and setup score.
- Bounce Short: compares current moves against historical dollar-block resistance zones.
- First Red Day: looks for multi-day parabolic runs, exhaustion risk, and first-red-day triggers.
- Watchlist: monitors manually entered tickers with crack/rejection/volume-fade status and a chart.

When paid APIs are not configured, the scanner uses Yahoo/fallback mock data so the dashboard remains usable for testing. It does not place trades.

## Options Paper Trading

Options support is single-leg and paper-only. The options scanner maps bullish stock signals to call candidates and bearish/avoid signals to put candidates, then ranks active tradable contracts by DTE, moneyness, underlying signal strength, suitability, and risk/reward. The order page submits Alpaca paper limit orders only; no live options trading is implemented.

Buy orders require the option premium times 100 times contracts. Sell put orders require cash-secured collateral based on strike times 100 times contracts, so the app checks available options buying power before submit. Selling uncovered calls is disabled.

The Options Paper Trading page includes a Wheel Strategy tab. It scans for cash-secured put candidates using a default target of 30 DTE, a DTE range of 14-45 days, target absolute delta of 0.15, and delta range of 0.10-0.20. Alpaca option snapshots are used for delta when available; otherwise the app labels the delta as estimated.

The stock paper bot and options pages can use separate Alpaca paper accounts:

```toml
ALPACA_API_KEY="stock_bot_paper_key"
ALPACA_SECRET_KEY="stock_bot_paper_secret"
ALPACA_ENDPOINT="https://paper-api.alpaca.markets/v2"

ALPACA_OPTIONS_API_KEY="options_paper_key"
ALPACA_OPTIONS_SECRET_KEY="options_paper_secret"
ALPACA_OPTIONS_ENDPOINT="https://paper-api.alpaca.markets/v2"
```

## Project Structure

```text
streamlit_app.py
requirements.txt
.env.example
README.md
app/
  core/
    analyzer.py
    backtesting.py
    config.py
    data.py
    indicators.py
    modeling.py
    paper_trading.py
    screener.py
    signals.py
    suitability.py
    watchlists.py
  storage/
logs/
```
