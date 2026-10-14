"""Client for the OpenRouter Decisions API (POST /api/alpha/decisions).

"Decision" models on OpenRouter (output_modalities=decisions) do not accept
chat/completions: they answer structured questions over a JSON `state` and
return typed choices. This module wraps that endpoint and maps the answers
to vending-machine price actions.
"""
import json
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


def build_decision_questions(product_names):
    """One `choice` question per product over the symmetric price-action menu."""
    criteria = {k: desc for k, (desc, _) in PRICE_ACTION_MENU.items()}
    questions = {}
    for name in product_names:
        questions[_question_key(name)] = {
            "type": "choice",
            "instructions": (
                f"Pricing decision for product '{name}' of LLMMachine for next week. "
                "Your goal is to maximize NET PROFIT (margin x volume), not sales volume. "
                "BasicMachine keeps its prices static; you share the same client pool and clients "
                "prefer cheaper products. If you sold out last week, there was unmet demand you can "
                "monetize with higher prices; if stock was left over, consider lower prices or better margins."
            ),
            "criteria": criteria,
        }
    return questions


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
