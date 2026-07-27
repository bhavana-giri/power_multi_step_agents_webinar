"""Live operational data + Context Retriever-style tools for the Iris mode.

Emulates the shape of Redis Context Retriever: declared data models over Redis
(inventory, orders) exposed to the agent as typed tools. The data genuinely
lives in and is queried from Redis, so the demo shows memory (Agent Memory)
and live business data (Redis) fused into one prompt — the deck's Part 05.
Swap these handlers for the managed Context Retriever MCP tools to use the
real service.
"""

from __future__ import annotations

import re
from typing import Any

import redis

from backend.catalog import CATALOG
from backend.settings import Settings

SEED_MARKER = "live:seeded:v1"

# Live overlay on the static catalog: stock, discounts, delivery ETA.
# RAV4 (the demo's "top pick") is in stock with a small discount; a couple of
# models are out of stock so availability questions have interesting answers.
LIVE_INVENTORY = {
    "Toyota RAV4 Hybrid": {"stock": 2, "discount_pct": 3, "eta_days": 0},
    "Honda CR-V Hybrid": {"stock": 5, "discount_pct": 0, "eta_days": 0},
    "Hyundai Tucson Hybrid": {"stock": 4, "discount_pct": 7, "eta_days": 0},
    "Ford Escape Hybrid": {"stock": 0, "discount_pct": 0, "eta_days": 14},
    "Kia Sportage Hybrid": {"stock": 0, "discount_pct": 0, "eta_days": 21},
    "Toyota Highlander Hybrid": {"stock": 3, "discount_pct": 0, "eta_days": 0},
    "Tesla Model Y": {"stock": 6, "discount_pct": 5, "eta_days": 0},
    "Mazda CX-50": {"stock": 7, "discount_pct": 0, "eta_days": 0},
    "Toyota Camry Hybrid": {"stock": 8, "discount_pct": 4, "eta_days": 0},
    "Honda Civic": {"stock": 10, "discount_pct": 0, "eta_days": 0},
}

# Order 4471 deliberately matches the example on the deck's
# "Memory recalls; it can't look things up" slide.
ORDERS = [
    {
        "order_id": "4471",
        "user_id": "demo-user",
        "item": "Roof rack + crossbars",
        "status": "Shipped",
        "eta": "Arriving Jul 30",
    },
    {
        "order_id": "4502",
        "user_id": "demo-user",
        "item": "Test drive: Toyota RAV4 Hybrid",
        "status": "Scheduled",
        "eta": "Aug 2, 10:00 AM",
    },
]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_inventory",
            "description": (
                "Search live vehicle inventory. Returns current stock, list price, "
                "discount, and delivery ETA per model. Filter by fuel, body type, "
                "max price, or in-stock only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fuel": {"type": "string", "enum": ["hybrid", "electric", "gas"]},
                    "body_type": {"type": "string", "enum": ["SUV", "sedan"]},
                    "max_price": {"type": "number"},
                    "in_stock_only": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle",
            "description": (
                "Get live stock, price, discount, and delivery ETA for one model "
                "by (partial) name."
            ),
            "parameters": {
                "type": "object",
                "properties": {"model": {"type": "string"}},
                "required": ["model"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": (
                "Get this customer's orders and scheduled appointments with "
                "current status."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_client: redis.Redis | None = None


def _redis(settings: Settings) -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def seed(settings: Settings) -> bool:
    """Idempotently load live inventory + orders into Redis."""
    r = _redis(settings)
    if r.exists(SEED_MARKER):
        return False
    for car in CATALOG:
        live = LIVE_INVENTORY.get(car["model"], {"stock": 3, "discount_pct": 0, "eta_days": 0})
        r.hset(
            f"live:inv:{_slug(car['model'])}",
            mapping={
                "model": car["model"],
                "type": car["type"],
                "fuel": car["fuel"],
                "price_usd": car["price_usd"],
                "mpg": car["mpg"],
                "seats": car["seats"],
                **live,
            },
        )
    for order in ORDERS:
        r.hset(f"live:order:{order['order_id']}", mapping=order)
    r.set(SEED_MARKER, "1")
    return True


def _all_inventory(settings: Settings) -> list[dict[str, Any]]:
    r = _redis(settings)
    rows = []
    for key in sorted(r.scan_iter("live:inv:*", count=100)):
        row = r.hgetall(key)
        for field in ("price_usd", "mpg", "seats", "stock", "discount_pct", "eta_days"):
            row[field] = int(row.get(field, 0))
        rows.append(row)
    return rows


def search_inventory(settings: Settings, args: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _all_inventory(settings)
    if args.get("fuel"):
        rows = [x for x in rows if x["fuel"] == args["fuel"]]
    if args.get("body_type"):
        rows = [x for x in rows if x["type"] == args["body_type"]]
    if args.get("max_price") is not None:
        rows = [x for x in rows if x["price_usd"] <= args["max_price"]]
    if args.get("in_stock_only"):
        rows = [x for x in rows if x["stock"] > 0]
    return rows


def get_vehicle(settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    needle = str(args.get("model", "")).lower()
    for row in _all_inventory(settings):
        if needle in row["model"].lower():
            return row
    return {"error": f"No model matching '{args.get('model')}' in inventory."}


def get_customer_orders(settings: Settings, args: dict[str, Any]) -> list[dict[str, Any]]:
    r = _redis(settings)
    orders = []
    for key in sorted(r.scan_iter("live:order:*", count=100)):
        order = r.hgetall(key)
        if order.get("user_id") == settings.demo_user_id:
            orders.append(order)
    return orders


_HANDLERS = {
    "search_inventory": search_inventory,
    "get_vehicle": get_vehicle,
    "get_customer_orders": get_customer_orders,
}


def execute_tool(settings: Settings, name: str, args: dict[str, Any]) -> Any:
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    seed(settings)
    return handler(settings, args)
