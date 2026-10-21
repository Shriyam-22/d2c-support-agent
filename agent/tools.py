"""Tools the agent can call.

Each tool has a Pydantic model for its arguments. The model does three jobs:
  1. generates the JSON schema sent to the LLM for function calling
  2. validates the arguments the LLM produces before anything runs
  3. documents the tool for humans

Tool results are plain dicts, serialised to JSON for the LLM. Errors are
returned as {"error": ...} instead of raised, so the model can recover
(e.g. ask the customer for a missing order ID) rather than the turn crashing.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from .retrieval import Retriever
from .store import OrderStore

# ---------------------------------------------------------------- argument models


class TrackOrderArgs(BaseModel):
    """Look up the status of a customer's order. Requires the order ID and the
    last 4 digits of the phone number used on the order, for verification."""

    order_id: str = Field(description="Order ID, e.g. NN1002. A leading # is fine.")
    phone_last4: str = Field(pattern=r"^\d{4}$",
                             description="Last 4 digits of the phone number on the order.")


class SearchProductsArgs(BaseModel):
    """Search the product catalogue for recommendations. Use for any question
    about which product to buy, product details, ingredients, sizes or prices."""

    query: str = Field(description="What the customer is looking for, e.g. 'face wash for oily skin'.")
    max_price: int | None = Field(default=None, ge=0, description="Optional budget limit in rupees.")
    category: Literal["skincare", "haircare", "bodycare", "kits"] | None = Field(
        default=None, description="Optional category filter.")


class SearchPoliciesArgs(BaseModel):
    """Search store policies: shipping, delivery times, COD, returns, exchanges,
    refunds, cancellation, address changes and product safety."""

    query: str = Field(description="The policy question, e.g. 'can I return an opened shampoo'.")


class EscalateArgs(BaseModel):
    """Hand the conversation to a human agent. Use when the customer asks for a
    human, is upset, reports a damaged/wrong item, asks for medical advice, or
    the request cannot be handled with the other tools."""

    reason: str = Field(description="Short summary of the issue for the human agent.")
    priority: Literal["low", "normal", "high"] = "normal"


# ---------------------------------------------------------------- registry


@dataclass
class Tool:
    name: str
    args_model: type[BaseModel]
    fn: Callable[[BaseModel], dict]

    def schema(self) -> dict:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        params.pop("description", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (self.args_model.__doc__ or "").strip().replace("\n", " "),
                "parameters": params,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, raw_arguments: str | dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            data = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            args = tool.args_model.model_validate(data or {})
        except (json.JSONDecodeError, ValidationError) as exc:
            return {"error": f"Invalid arguments for {name}: {exc}".splitlines()[0]
                    + " Ask the customer for the missing or correct details."}
        try:
            return tool.fn(args)
        except Exception as exc:  # a tool bug should not crash the conversation
            return {"error": f"{name} failed: {type(exc).__name__}. Apologise and offer a human agent."}


# ---------------------------------------------------------------- implementations

PUBLIC_ORDER_FIELDS = ("order_id", "status", "items", "total", "payment", "placed_on",
                       "carrier", "tracking_number", "expected_delivery", "last_update")


def build_registry(store: OrderStore, retriever: Retriever) -> ToolRegistry:
    ticket_counter = itertools.count(1001)

    def track_order(args: TrackOrderArgs) -> dict:
        order = store.get_order(args.order_id)
        if order is None:
            return {"found": False, "message": "No order with this ID. Ask the customer to re-check it."}
        # Verification guardrail: never reveal order details without a phone match.
        if not order["phone"].endswith(args.phone_last4):
            return {"found": True, "verified": False,
                    "message": "Phone digits do not match this order. Do not share any order details."}
        return {"found": True, "verified": True, **{k: order[k] for k in PUBLIC_ORDER_FIELDS}}

    def search_products(args: SearchProductsArgs) -> dict:
        hits = retriever.search(args.query, k=8, kind="product")
        products = []
        for h in hits:
            p = h.chunk.metadata
            if args.category and p["category"] != args.category:
                continue
            if args.max_price is not None and p["price"] > args.max_price:
                continue
            products.append({k: p[k] for k in ("sku", "name", "category", "price", "size", "in_stock", "description")})
        if not products:
            return {"products": [], "message": "No matching products. Say so honestly; do not invent products."}
        return {"products": products[:3]}

    def search_policies(args: SearchPoliciesArgs) -> dict:
        hits = retriever.search(args.query, k=2, kind="policy")
        if not hits:
            return {"policies": [], "message": "No matching policy. Offer to connect a human agent."}
        return {"policies": [{"title": h.chunk.metadata["title"], "text": h.chunk.text} for h in hits]}

    def escalate_to_human(args: EscalateArgs) -> dict:
        ticket = f"ESC-{next(ticket_counter)}"
        return {"ticket_id": ticket, "priority": args.priority,
                "message": "A human agent will reply on this chat within 2 working hours."}

    return ToolRegistry([
        Tool("track_order", TrackOrderArgs, track_order),
        Tool("search_products", SearchProductsArgs, search_products),
        Tool("search_policies", SearchPoliciesArgs, search_policies),
        Tool("escalate_to_human", EscalateArgs, escalate_to_human),
    ])
