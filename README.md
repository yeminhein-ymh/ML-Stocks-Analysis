# Universal AI Stock Analysis Dashboard

Streamlit dashboard and paper-trading bot for universal multi-stock analysis across liquid US stocks. It is no longer limited to the original fixed ticker set.

## Features

- Manual entry for any US ticker.
- Multi-select scanning for 5, 10, 20, 50, or 100 symbols.
- Default watchlist: NVDA, AMZN, META, GOOGL, PLTR, MU, SNDK, NBIS, AAPL, MSFT, TSLA, AMD, AVGO, SMCI, NFLX, ARM, TSM, ORCL, CRM, QCOM, INTC, COIN, MSTR, SOFI, HOOD, JPM, BAC, XOM, NEE, ENPH, FSLR.
- CSV watchlist upload and saved custom watchlists.
- Stock screener for formula suitability.
- Suitability gate before AI signals.
- Random Forest and optional XGBoost training with time-series splits.
- Individual and portfolio backtesting with cost and slippage.
- Alpaca paper-trading helpers only. No live trading implementation.

## Pages

1. Market Overview
2. Multi-Stock AI Scanner
3. Individual Stock Analysis
4. Technical Indicator Dashboard
5. AI Prediction Model
6. Backtesting
7. Paper Trading Bot
8. Risk Management
9. Portfolio Allocation
10. Watchlist Manager

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
