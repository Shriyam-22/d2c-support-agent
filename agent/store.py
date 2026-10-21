"""Order data access.

`OrderStore` is the interface the agent's tools depend on. `MockShopifyStore`
reads orders from data/orders.json so the whole project runs offline.
A real Shopify Admin API client can implement the same `get_order` method
without any change to the agent, tools or evals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class OrderStore(Protocol):
    def get_order(self, order_id: str) -> dict | None: ...


def normalise_order_id(order_id: str) -> str:
    return order_id.strip().upper().lstrip("#").replace(" ", "")


class MockShopifyStore:
    def __init__(self, path: Path):
        orders = json.loads(path.read_text(encoding="utf-8"))
        self._orders = {normalise_order_id(o["order_id"]): o for o in orders}

    def get_order(self, order_id: str) -> dict | None:
        return self._orders.get(normalise_order_id(order_id))
