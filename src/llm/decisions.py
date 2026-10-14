"""Client for the OpenRouter Decisions API (POST /api/alpha/decisions).

"Decision" models on OpenRouter (output_modalities=decisions) do not accept
chat/completions: they answer structured questions over a JSON `state` and
return typed choices. This module wraps that endpoint and maps the answers
to vending-machine price actions.
"""
import json
import os
import time
import urllib.request

from src.config import OPENROUTER_API_KEY, OPENROUTER_DECISIONS_URL

# Relative price-action menu offered to decision models (chat models choose
# any price freely; decision models pick from a symmetric action menu).
PRICE_ACTION_MENU = {
    "decrease_large": ("Cut the price by 15% (p * 0.85). Lowest margin per sale, most likely to win clients.", 0.85),
    "decrease_small": ("Cut the price by 7.5% (p * 0.925). Small margin sacrifice to gain volume.", 0.925),
    "keep": ("Keep the current price unchanged.", 1.0),
    "increase_small": ("Raise the price by 7.5% (p * 1.075). Higher margin per sale, slightly fewer sales.", 1.075),
    "increase_large": ("Raise the price by 15% (p * 1.15). Much higher margin per sale, likely fewer sales.", 1.15),
}


def _question_key(product_name: str) -> str:
    return product_name.lower().replace(" ", "_")


def product_from_question_key(key: str, product_names):
    for name in product_names:
        if _question_key(name) == key:
            return name
    return None


def build_product_question(product_name):
    """One focused `choice` question per product over the price-action menu.

    Decision models are classification-style models: they answer best with a
    single focused question and a flat product-scoped state (see
    build_product_state), not a big nested JSON blob.
    """
    criteria = {k: desc for k, (desc, _) in PRICE_ACTION_MENU.items()}
    return {
        _question_key(product_name): {
            "type": "choice",
            "instructions": (
                f"Pricing decision for product '{product_name}' of LLMMachine for next week. "
                "Your goal is to maximize NET PROFIT (margin x volume), not sales volume. "
                "BasicMachine keeps its prices static; you share the same client pool and clients "
                "prefer cheaper products. A stockout means unmet demand you can monetize with higher "
                "prices; leftover stock means demand is weak at the current price."
            ),
            "criteria": criteria,
        }
    }


def build_product_state(product_name, market_data, stockout_events, week):
    """Flat, product-scoped state for one pricing decision."""
    llm_products = market_data.get("LLMMachine", {}).get("products", {})
    basic_products = market_data.get("BasicMachine", {}).get("products", {})
    product = llm_products[product_name]
    competitor = basic_products.get(product_name, {})
    stock = product.get("stock", 0)
    max_stock = product.get("max_stock", 0)
    if stock == 0:
        stock_desc = f"0 of {max_stock} (SOLD OUT: unmet demand last week)"
    else:
        stock_desc = f"{stock} of {max_stock} left"
    relevant_stockouts = [e for e in (stockout_events or []) if product_name.lower() in e.lower()]
    return {
        "week": str(week),
        "product": product_name,
        "your_current_price": str(product.get("price")),
        "unit_cost": str(product.get("cost")),
        "competitor_price": str(competitor.get("price")),
        "stock_left": stock_desc,
        "purchase_likelihood_at_current_price": str(product.get("likelihood")),
        "your_machine_total_profit": str(market_data.get("LLMMachine", {}).get("profit_loss")),
        "competitor_machine_total_profit": str(market_data.get("BasicMachine", {}).get("profit_loss")),
        "stockouts_last_week": "; ".join(relevant_stockouts) if relevant_stockouts else "none",
    }


def call_decision_model(model, questions, state, max_retries=4, timeout=120.0):
    """POST to the decisions endpoint. Returns the parsed `answers` dict."""
    body = {"model": model, "questions": questions, "state": state}
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                OPENROUTER_DECISIONS_URL,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
            answers = payload.get("answers")
            if answers is None:
                raise ValueError(f"No 'answers' in response: {str(payload)[:200]}")
            return answers
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"[Decisions] Call failed ({str(e)[:140]}). Retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"Decisions API failed after {max_retries} attempts: {last_err}")


def decide_prices_for_week(model, market_data, stockout_events, week, verbose=False):
    """Make one pricing decision per product (one focused API call each).

    Returns a combined answers dict keyed by question key. Products whose call
    fails after retries are omitted (price kept) so one flaky call cannot
    derail the whole week.
    """
    llm_products = list(market_data.get("LLMMachine", {}).get("products", {}).keys())
    answers = {}
    for name in llm_products:
        questions = build_product_question(name)
        state = build_product_state(name, market_data, stockout_events, week)
        if os.getenv("DEBUG_DECISIONS"):
            print(f"[DEBUG] {model} | {name} question={json.dumps(questions)} state={json.dumps(state)}")
        try:
            answer = call_decision_model(model, questions, state)
            answers.update(answer)
        except Exception as e:  # noqa: BLE001
            print(f"[Decisions] {name} decision failed, keeping price: {str(e)[:140]}")
    return answers


def answers_to_price_actions(answers, market_data):
    """Map decision answers to change_price parameter dicts.

    Returns (actions, reasons) where actions is a list of change_price kwargs.
    Invalid/missing answers are skipped (price kept).
    """
    llm_products = market_data.get("LLMMachine", {}).get("products", {})
    actions = []
    reasons = []
    for key, answer in (answers or {}).items():
        product = product_from_question_key(key, llm_products.keys())
        if not product:
            continue
        choice = answer.get("choice") if isinstance(answer, dict) else None
        if choice not in PRICE_ACTION_MENU:
            continue
        _, factor = PRICE_ACTION_MENU[choice]
        current = llm_products[product]["price"]
        new_price = round(current * factor, 2)
        if abs(new_price - current) < 0.005:
            continue  # keep: no action needed
        actions.append({
            "machine_name": "LLMMachine",
            "product_name": product,
            "new_price": new_price,
        })
        confidence = answer.get("confidence")
        reasons.append(f"{product}: {choice} (p={current} -> {new_price}, confidence={confidence})")
    return actions, reasons
