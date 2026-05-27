from __future__ import annotations

import os
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
