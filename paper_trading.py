from __future__ import annotations

import os
import re
from datetime import datetime

import pandas as pd

from app.core.config import RISK_RULES, SIGNAL_LOG_FILE, TRADE_LOG_FILE

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.common.exceptions import APIError
except Exception:
    TradingClient = None
    GetOrdersRequest = None
    LimitOrderRequest = None
    MarketOrderRequest = None
    OrderSide = None
    QueryOrderStatus = None
    TimeInForce = None
    APIError = Exception


def alpaca_client():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    paper_endpoint = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    if not key or not secret or TradingClient is None:
        return None
    sdk_base_url = paper_endpoint.removesuffix("/").removesuffix("/v2")
    return TradingClient(key, secret, paper=True, url_override=sdk_base_url)


def append_log(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**{"timestamp": datetime.utcnow().isoformat()}, **row}])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def position_size(capital: float, entry: float, stop: float, current_exposure: float = 0) -> int:
    risk_cap = capital * RISK_RULES["max_risk_per_trade"]
    risk_per_share = max(0.01, entry - stop)
    shares_by_risk = int(risk_cap / risk_per_share)
    shares_by_exposure = int((capital * RISK_RULES["max_exposure_per_stock"] - current_exposure) / entry)
    return max(0, min(shares_by_risk, shares_by_exposure))


def parse_option_symbol(option_symbol: str) -> dict:
    match = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", option_symbol.strip().upper())
    if not match:
        return {}
    root, expiry, option_type, strike_raw = match.groups()
    return {
        "underlying": root,
        "expiration": f"20{expiry[:2]}-{expiry[2:4]}-{expiry[4:6]}",
        "type": "call" if option_type == "C" else "put",
        "strike": int(strike_raw) / 1000,
    }


def account_buying_power(client) -> float:
    try:
        account = client.get_account()
    except Exception:
        return 0.0
    for field in ("options_buying_power", "buying_power", "cash"):
        value = getattr(account, field, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def option_order_requirement(option_symbol: str, qty: int, limit_price: float, side: str) -> dict:
    details = parse_option_symbol(option_symbol)
    if qty <= 0:
        return {"ok": False, "required": 0.0, "message": "Quantity must be greater than zero."}
    if limit_price <= 0:
        return {"ok": False, "required": 0.0, "message": "Limit price must be greater than zero."}
    if side.lower() == "buy":
        return {"ok": True, "required": qty * limit_price * 100, "message": "Long option premium required."}
    if not details:
        return {"ok": False, "required": 0.0, "message": "Could not parse option symbol for sell-side collateral check."}
    if details["type"] == "call":
        return {
            "ok": False,
            "required": 0.0,
            "message": "Selling uncovered calls is disabled in this paper bot. Use buy orders or close an existing long call manually.",
        }
    return {
        "ok": True,
        "required": details["strike"] * 100 * qty,
        "message": "Cash-secured put collateral required.",
    }


def open_orders_for_symbol(client, ticker: str) -> list:
    if GetOrdersRequest is None:
        return []
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
    return list(client.get_orders(filter=request))


def opposite_side_orders(client, ticker: str, side: str) -> list:
    desired_side = side.lower()
    opposite = "sell" if desired_side == "buy" else "buy"
    orders = open_orders_for_symbol(client, ticker)
    return [order for order in orders if str(getattr(order, "side", "")).lower().endswith(opposite)]


def open_orders_table(ticker: str | None = None) -> pd.DataFrame:
    client = alpaca_client()
    if client is None:
        return pd.DataFrame()
    if GetOrdersRequest is None:
        return pd.DataFrame()
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker] if ticker else None)
    rows = []
    for order in client.get_orders(filter=request):
        rows.append(
            {
                "id": str(getattr(order, "id", "")),
                "symbol": getattr(order, "symbol", ""),
                "side": str(getattr(order, "side", "")).replace("OrderSide.", "").lower(),
                "qty": getattr(order, "qty", ""),
                "type": str(getattr(order, "type", "")).replace("OrderType.", "").lower(),
                "status": str(getattr(order, "status", "")).replace("OrderStatus.", "").lower(),
                "submitted_at": getattr(order, "submitted_at", ""),
            }
        )
    return pd.DataFrame(rows)


def cancel_order(order_id: str) -> dict:
    client = alpaca_client()
    if client is None:
        return {"ok": False, "message": "Alpaca paper API is not configured."}
    try:
        client.cancel_order_by_id(order_id)
    except APIError as exc:
        return {"ok": False, "message": f"Could not cancel order: {exc}"}
    return {"ok": True, "message": f"Canceled paper order {order_id}."}


def place_paper_order(ticker: str, qty: int, side: str = "buy") -> dict:
    client = alpaca_client()
    if client is None:
        return {"ok": False, "message": "Alpaca paper API is not configured."}
    if qty <= 0:
        return {"ok": False, "message": "Quantity is zero after risk controls."}
    blockers = opposite_side_orders(client, ticker, side)
    if blockers:
        order_ids = ", ".join(str(getattr(order, "id", "unknown")) for order in blockers)
        return {
            "ok": False,
            "message": (
                f"Alpaca blocked this {side} for {ticker} because open opposite-side order(s) exist: "
                f"{order_ids}. Cancel or wait for those paper orders before submitting a new opposite order."
            ),
        }
    order = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    try:
        submitted = client.submit_order(order)
    except APIError as exc:
        return {"ok": False, "message": f"Alpaca rejected the paper order: {exc}"}
    append_log(TRADE_LOG_FILE, {"ticker": ticker, "qty": qty, "side": side, "alpaca_id": submitted.id})
    return {"ok": True, "message": f"Submitted paper {side} order for {qty} {ticker}."}


def place_option_paper_order(option_symbol: str, qty: int, limit_price: float, side: str = "buy") -> dict:
    client = alpaca_client()
    if client is None:
        return {"ok": False, "message": "Alpaca paper API is not configured."}
    if LimitOrderRequest is None:
        return {"ok": False, "message": "Installed alpaca-py version does not support limit order requests."}
    if qty <= 0:
        return {"ok": False, "message": "Quantity is zero."}
    if limit_price <= 0:
        return {"ok": False, "message": "Limit price must be greater than zero."}
    requirement = option_order_requirement(option_symbol, qty, limit_price, side)
    if not requirement["ok"]:
        return {"ok": False, "message": requirement["message"]}
    available = account_buying_power(client)
    if requirement["required"] > available:
        return {
            "ok": False,
            "message": (
                f"Insufficient options buying power. Required about ${requirement['required']:,.2f}; "
                f"available about ${available:,.2f}. For sell puts, Alpaca requires cash-secured collateral "
                "based on strike x 100 x contracts, not the limit premium."
            ),
        }
    blockers = opposite_side_orders(client, option_symbol, side)
    if blockers:
        order_ids = ", ".join(str(getattr(order, "id", "unknown")) for order in blockers)
        return {"ok": False, "message": f"Open opposite-side option order exists: {order_ids}. Cancel it before submitting."}
    order = LimitOrderRequest(
        symbol=option_symbol.strip().upper(),
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=round(float(limit_price), 2),
    )
    try:
        submitted = client.submit_order(order)
    except APIError as exc:
        return {"ok": False, "message": f"Alpaca rejected the option paper order: {exc}"}
    append_log(
        TRADE_LOG_FILE,
        {"ticker": option_symbol.strip().upper(), "qty": qty, "side": side, "asset_class": "option", "limit_price": limit_price, "alpaca_id": submitted.id},
    )
    return {"ok": True, "message": f"Submitted paper option {side} limit order for {qty} {option_symbol} at ${limit_price:.2f}."}


def log_signal(row: dict) -> None:
    append_log(SIGNAL_LOG_FILE, row)
